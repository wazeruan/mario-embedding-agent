import copy

import pytest
import torch
from torch.nn import functional as F

from mario_embedding_agent.actions import ACTIONS
from mario_embedding_agent.policy import ActorCritic
from mario_embedding_agent.ppo import Transition, update


def vectors():
    return torch.randn(len(ACTIONS), 768, generator=torch.Generator().manual_seed(31))


def test_mlp_zero_residual_preserves_preencoded_prior_and_gradients_reach_output():
    torch.set_num_threads(1)
    inputs, previous = torch.randn(3, 768), torch.randn(3, 768)
    legacy = ActorCritic(vectors())
    mlp = ActorCritic(vectors(), architecture='mlp', feature_mean=torch.full((768,), .1), feature_std=torch.full((768,), .03))
    a, b = legacy(inputs, previous_features=previous), mlp(inputs, previous_features=previous)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    assert mlp.nonlinear_actor[1].in_features == 1536
    (b[0][:, 0].sum() + b[1].sum()).backward()
    assert mlp.nonlinear_actor[-1].weight.grad.norm() > 0.
    assert mlp.critic[-1].weight.grad.norm() > 0.


def test_mlp_joint_context_is_nonlinear_and_batches_match_single_evaluation():
    model = ActorCritic(vectors(), architecture='mlp')
    with torch.no_grad():
        model.nonlinear_actor[-1].weight.normal_(std=.1)
        model.critic[-1].weight.normal_(std=.1)
    current, previous = torch.randn(3, 768), torch.randn(3, 768)
    batch_logits, batch_values = model(current, previous_features=previous)
    for i in range(3):
        logits, value = model(current[i], previous_features=previous[i])
        torch.testing.assert_close(logits, batch_logits[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(value, batch_values[i], rtol=1e-5, atol=1e-5)
    s1, s2, d1, d2 = torch.randn(4, 768)
    def residual(state, delta):
        return model.nonlinear_actor(torch.cat((state, delta)))
    interaction = residual(s1, d1)+residual(s2, d2)-residual(s1, d2)-residual(s2, d1)
    assert interaction.norm() > .01  # Joint nonlinear interaction, beyond additive branches.


def test_statistics_whiten_current_and_delta_with_same_fixed_scale_and_roundtrip(tmp_path):
    mean, std = torch.full((768,), .1), torch.full((768,), .03)
    model = ActorCritic(vectors(), architecture='mlp', feature_mean=mean, feature_std=std)
    mean.zero_(); std.fill_(1.)  # Model owns independent copies.
    captured = []
    handle = model.nonlinear_actor.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    current, previous = torch.randn(768), torch.randn(768)
    model(current, previous_features=previous)
    handle.remove()
    normalized = F.normalize(current, dim=0)
    expected_current = ((normalized-.1)/.03).clamp(-10, 10)
    expected_delta = ((normalized-F.normalize(previous, dim=0))/.03).clamp(-10, 10)
    torch.testing.assert_close(captured[0], torch.cat((expected_current, expected_delta)))
    path = tmp_path/'policy.pt'
    torch.save(model.state_dict(), path)
    restored = ActorCritic(vectors(), architecture='mlp')
    restored.load_state_dict(torch.load(path, weights_only=True))
    for actual, expected in zip(restored(current, previous_features=previous), model(current, previous_features=previous)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('bad', ['shape', 'mean_nan', 'std_inf', 'std_zero', 'std_negative', 'missing_std'])
def test_mlp_statistics_validation(bad):
    mean, std = torch.zeros(768), torch.ones(768)
    if bad == 'shape': mean = torch.zeros(767)
    elif bad == 'mean_nan': mean[0] = float('nan')
    elif bad == 'std_inf': std[0] = float('inf')
    elif bad == 'std_zero': std[0] = 0.
    elif bad == 'std_negative': std[0] = -1.
    elif bad == 'missing_std': std = None
    with pytest.raises(ValueError):
        ActorCritic(vectors(), architecture='mlp', feature_mean=mean, feature_std=std)


def test_ppo_trains_new_actor_residual_and_value_head():
    torch.manual_seed(119)
    model = ActorCritic(vectors(), architecture='mlp', feature_mean=torch.zeros(768), feature_std=torch.full((768,), .04))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    transitions = []
    for i in range(8):
        current, previous = torch.randn(768), torch.randn(768)
        action, log_prob, value, _ = model.act(current, previous_features=previous)
        transitions.append(Transition(current, action, log_prob, value, float(i % 3), i == 7, previous_features=previous))
    actor_before = model.nonlinear_actor[-1].weight.detach().clone()
    critic_before = model.critic[-1].weight.detach().clone()
    result = update(model, optimizer, transitions, 0., epochs=2, minibatch_size=8, target_kl=None)
    assert result['actor_samples'] == 8 and result['optimizer_steps'] == 2
    assert not torch.equal(actor_before, model.nonlinear_actor[-1].weight)
    assert not torch.equal(critic_before, model.critic[-1].weight)


def test_assisted_mlp_actor_keeps_all_weights_and_seeded_adam_moments():
    model = ActorCritic(vectors(), architecture='mlp')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    actor = list(model.nonlinear_actor.parameters())
    before = [parameter.detach().clone() for parameter in actor]
    moments = [copy.deepcopy(optimizer.state[parameter]) for parameter in actor]
    bias = torch.zeros(len(ACTIONS)); bias[5] = 8.
    transitions = []
    for i in range(8):
        current = torch.randn(768)
        action, log_prob, value, _ = model.act(current, action_bias=bias)
        transitions.append(Transition(current, action, log_prob, value, 1., i == 7, bias))
    update(model, optimizer, transitions, 0., epochs=1, minibatch_size=8)
    for parameter, previous, old_moments in zip(actor, before, moments):
        torch.testing.assert_close(parameter, previous, rtol=0., atol=0.)
        for key, value in old_moments.items():
            torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0., atol=0.)


def test_calibration_uses_normalized_actual_student_outputs_on_training_streams_only(tmp_path):
    from test_student import dataset
    from mario_embedding_agent.student import calibrate_policy_features
    path = tmp_path/'frames.pt'
    _, groups = dataset(path)
    class Encoder:
        calls = []
        class Model:
            target_mean = torch.full((768,), 99.)
            target_std = torch.full((768,), 77.)
        model = Model()
        def encode_frame(self, frame):
            index = int(frame[0, 0, 0])
            self.calls.append(index)
            values = torch.arange(1, 769).float()
            values[:256] += index*10.
            return F.normalize(values, dim=0)
    encoder = Encoder()
    mean, std = calibrate_policy_features(encoder, path)
    train_indices = [i for i, group in enumerate(groups) if group[0] != 3]
    assert encoder.calls == train_indices
    expected = torch.stack([encoder.encode_frame(torch_frame.numpy())
                            for torch_frame in torch.load(path, weights_only=True)['frames'][groups[:, 0] != 3]])
    torch.testing.assert_close(mean, expected.mean(0))
    torch.testing.assert_close(std, expected.std(0).clamp_min(1e-3))
    assert not torch.equal(mean, encoder.model.target_mean)
    assert not torch.equal(std, encoder.model.target_std)


def test_joint_critic_receives_current_delta_gradients_and_reward_scale_reset():
    from mario_embedding_agent.cli import reset_critic_value_scale
    model = ActorCritic(vectors(), architecture='mlp', feature_mean=torch.full((768,), .01), feature_std=torch.full((768,), .04))
    assert model.joint_critic[1].in_features == 1536
    inputs = torch.randn(768)
    previous = torch.randn(768)
    assert model(inputs, previous_features=previous)[1] == 0.
    model(inputs, previous_features=previous)[1].backward()
    assert model.joint_critic[-1].weight.grad.norm() > 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    hidden_before = model.joint_critic[1].weight.detach().clone()
    assert any(parameter in optimizer.state for parameter in model.joint_critic.parameters())
    reset_critic_value_scale(model, optimizer)
    for head in (model.critic, model.temporal_critic, model.joint_critic):
        assert torch.count_nonzero(head[-1].weight) == 0
        assert torch.count_nonzero(head[-1].bias) == 0
        assert all(parameter not in optimizer.state for parameter in head.parameters())
    torch.testing.assert_close(model.joint_critic[1].weight, hidden_before)
    assert model.actor.weight in optimizer.state
