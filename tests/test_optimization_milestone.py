"""Running feature statistics, optional value clipping, unthrottled sync, and checkpoint migration."""
import copy
import json
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mario_embedding_agent.actions import ACTIONS
from mario_embedding_agent.policy import ActorCritic, CALIBRATION_PRIOR_COUNT
from mario_embedding_agent.ppo import Transition, update


def vectors():
    return torch.randn(len(ACTIONS), 768, generator=torch.Generator().manual_seed(3))


def population_stats(batch):
    return batch.mean(0), batch.var(0, unbiased=False).sqrt()


def test_running_statistics_from_empty_prior_equal_concatenated_batch_statistics():
    model = ActorCritic(vectors(), architecture='mlp')
    assert model.feature_count.dtype == torch.float64 and float(model.feature_count) == 0.
    generator = torch.Generator().manual_seed(0)
    # Unnormalized inputs: the policy normalizes before merging, like its forward pass.
    first = torch.randn(5, 768, generator=generator) * 3 + .5
    second = torch.randn(11, 768, generator=generator) * .2 - 1
    model.update_feature_statistics(first)
    mean, std = population_stats(F.normalize(first, dim=-1))
    torch.testing.assert_close(model.feature_mean, mean)
    torch.testing.assert_close(model.feature_std, std)
    model.update_feature_statistics(second)
    mean, std = population_stats(F.normalize(torch.cat((first, second)), dim=-1))
    torch.testing.assert_close(model.feature_mean, mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.feature_std, std, atol=1e-6, rtol=1e-5)
    assert float(model.feature_count) == 16.


def test_running_statistics_merge_with_nonzero_prior_count():
    generator = torch.Generator().manual_seed(1)
    prior = F.normalize(torch.randn(40, 768, generator=generator) + 1, dim=-1)
    batch = torch.randn(24, 768, generator=generator)
    prior_mean, prior_std = population_stats(prior)
    model = ActorCritic(vectors(), architecture='mlp', feature_mean=prior_mean,
                        feature_std=prior_std, feature_count=len(prior))
    model.update_feature_statistics(batch)
    mean, std = population_stats(torch.cat((prior, F.normalize(batch, dim=-1))))
    torch.testing.assert_close(model.feature_mean, mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.feature_std, std, atol=1e-6, rtol=1e-5)
    assert float(model.feature_count) == 64.


def test_calibrated_statistics_default_to_prior_count_and_dominate_small_rollout():
    mean, std = torch.full((768,), .02), torch.full((768,), .04)
    model = ActorCritic(vectors(), architecture='mlp', feature_mean=mean, feature_std=std)
    assert float(model.feature_count) == CALIBRATION_PRIOR_COUNT == 1024
    batch = F.normalize(torch.randn(16, 768, generator=torch.Generator().manual_seed(2)), dim=-1)
    model.update_feature_statistics(batch)
    assert float(model.feature_count) == 1040.
    torch.testing.assert_close(model.feature_mean, mean + (batch.mean(0) - mean) * 16 / 1040)


def test_feature_statistics_floor_std_and_ignore_empty_batch():
    model = ActorCritic(vectors(), architecture='mlp')
    constant = torch.ones(8, 768)
    model.update_feature_statistics(constant)
    assert torch.all(model.feature_std == 1e-3)
    torch.testing.assert_close(model.feature_mean, F.normalize(constant, dim=-1)[0])
    before = copy.deepcopy(model.state_dict())
    model.update_feature_statistics(torch.empty(0, 768))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_feature_statistics_are_noop_for_legacy_architecture():
    model = ActorCritic(vectors())
    before = copy.deepcopy(model.state_dict())
    model.update_feature_statistics(torch.randn(4, 768))
    assert not hasattr(model, 'feature_count') and not hasattr(model, 'feature_mean')
    assert model.state_dict().keys() == before.keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


class CriticOnly(torch.nn.Module):
    """Uniform fixed actor with a scalar value parameter, so critic loss is analytic."""

    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.actor = torch.nn.Linear(1, len(ACTIONS)).requires_grad_(False)  # Only out_features is read.

    def forward(self, features, action_bias=None, previous_features=None):
        batch = features.shape[:-1]
        return torch.zeros(*batch, len(ACTIONS)), self.value.expand(batch)


def run_critic_update(value_clip, start=0., reward=.6, old_value=0.):
    # Small errors keep the gradient below update()'s 0.5 global-norm clip.
    policy = CriticOnly(start)
    optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
    log_prob = float(-np.log(len(ACTIONS)))
    rollout = [Transition(torch.ones(768), 0, log_prob, old_value, reward, True) for _ in range(4)]
    kwargs = {} if value_clip == 'default' else {'value_clip': value_clip}
    metrics = update(policy, optimizer, rollout, 0., epochs=1, minibatch_size=4, entropy_coef=0., **kwargs)
    return float(policy.value.detach()), metrics


def test_value_clip_off_by_default_uses_plain_mse_gradient():
    # Plain loss value_coef*0.5*(v-R)^2 -> gradient 0.5*(v-R) = -0.3, SGD step +0.03.
    default, _ = run_critic_update('default')
    unclipped, _ = run_critic_update(None)
    assert default == unclipped == pytest.approx(.03)


def test_value_clip_uses_pessimistic_ppo2_max_and_differs_from_mse():
    # Value equals old value: unclipped and clipped errors coincide, so clipping changes nothing.
    assert run_critic_update(.2)[0] == pytest.approx(run_critic_update(None)[0])
    # Value already beyond old_value+clip: clipped term (0.2-0.6)^2 > (0.3-0.6)^2 dominates with zero gradient.
    clipped, _ = run_critic_update(.2, start=.3)
    unclipped, _ = run_critic_update(None, start=.3)
    assert clipped == pytest.approx(.3)
    assert unclipped == pytest.approx(.3 + .1 * .5 * .3)
    # A wide clip width behaves like plain MSE.
    assert run_critic_update(100., start=.3)[0] == pytest.approx(unclipped)


def test_load_policy_weights_migrates_missing_feature_count():
    from mario_embedding_agent.cli import load_policy_weights

    identity_source = ActorCritic(vectors(), architecture='mlp')
    weights = identity_source.state_dict()
    weights.pop('feature_count')
    target = ActorCritic(vectors(), architecture='mlp', feature_mean=torch.full((768,), .1),
                         feature_std=torch.full((768,), .5), feature_count=7)
    load_policy_weights(target, weights)
    assert float(target.feature_count) == 0.
    assert torch.all(target.feature_mean == 0) and torch.all(target.feature_std == 1)

    calibrated = ActorCritic(vectors(), architecture='mlp', feature_mean=torch.full((768,), .1),
                             feature_std=torch.full((768,), .05), feature_count=3)
    weights = calibrated.state_dict()
    weights.pop('feature_count')
    target = ActorCritic(vectors(), architecture='mlp', feature_count=0)
    load_policy_weights(target, weights)
    assert float(target.feature_count) == CALIBRATION_PRIOR_COUNT
    torch.testing.assert_close(target.feature_std, torch.full((768,), .05))

    # A present count is restored as saved, not re-derived.
    saved = ActorCritic(vectors(), architecture='mlp', feature_count=37).state_dict()
    target = ActorCritic(vectors(), architecture='mlp')
    load_policy_weights(target, saved)
    assert float(target.feature_count) == 37.

    # Other missing keys are still rejected.
    weights = identity_source.state_dict()
    weights.pop('feature_count')
    weights.pop('feature_mean')
    with pytest.raises(ValueError, match='policy keys are incompatible'):
        load_policy_weights(ActorCritic(vectors(), architecture='mlp'), weights)


def install_fake_cli(monkeypatch, tmp_path, cli, sleeps):
    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            generator = torch.Generator().manual_seed(int(frame[0, 0, 0]))
            return torch.randn(768, generator=generator)

    class Env:
        frames = 0
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    env = Env()
    def fake_step(environment, command):
        environment.frames += 1
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        frame[0, 0, 0] = environment.frames
        return frame, False
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    monkeypatch.setattr(cli.time, 'sleep', lambda seconds: sleeps.append(seconds))


@pytest.mark.parametrize('mode', ['realtime', None])
def test_fps_zero_rejected_outside_sync_mode(monkeypatch, tmp_path, capsys, mode):
    from mario_embedding_agent import cli
    install_fake_cli(monkeypatch, tmp_path, cli, [])
    argv = ['mario-agent', 'train', '--headless', '--fps', '0', '--total-frames', '4']
    if mode:
        argv += ['--mode', mode]  # Train otherwise defaults to realtime.
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert 'requires --mode sync' in capsys.readouterr().err


def test_negative_fps_still_rejected(monkeypatch, tmp_path, capsys):
    from mario_embedding_agent import cli
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync', '--fps', '-1'])
    with pytest.raises(SystemExit):
        cli.main()
    assert 'non-negative' in capsys.readouterr().err


def test_fps_zero_sync_runs_unthrottled_and_updates_feature_statistics_after_ppo(monkeypatch, tmp_path):
    from mario_embedding_agent import cli
    sleeps, calls, saved = [], [], []
    install_fake_cli(monkeypatch, tmp_path, cli, sleeps)

    def fake_update(policy, optimizer, rollout, bootstrap, **kwargs):
        assert kwargs['value_clip'] is None
        # Stored log-probabilities assume the statistics in force while acting.
        calls.append((float(policy.feature_count), policy.feature_mean.clone(),
                      torch.stack([item.features for item in rollout]).clone()))
        return {}

    def save(path, policy, optimizer, config, state):
        saved.append((copy.deepcopy(config), copy.deepcopy(policy.state_dict())))

    monkeypatch.setattr(cli, 'update', fake_update)
    monkeypatch.setattr(cli.checkpoint, 'save', save)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync', '--fps', '0',
                                    '--total-frames', '60', '--rollout', '2'])
    cli.main()

    assert sleeps == []
    assert len(calls) >= 2
    seen = []
    for index, (count, mean, features) in enumerate(calls):
        # A fresh run seeds statistics from its first rollout before the first update
        # (its zero-initialized heads ignore statistics); later rollouts merge afterwards.
        assert count == 2. * max(index, 1)
        if index == 0:
            torch.testing.assert_close(mean, F.normalize(features, dim=-1).mean(0))
        seen.append(features)
    config, weights = saved[-1]
    assert config['policy_architecture'] == 'mlp'
    assert config['feature_standardization'] == 'running-rollout-v1'
    assert config['value_clip'] is None
    assert float(weights['feature_count']) == 2. * len(calls)
    expected_mean, expected_std = population_stats(F.normalize(torch.cat(seen), dim=-1))
    # Many sequential float32 merges accumulate ~1e-4 relative rounding in std.
    torch.testing.assert_close(weights['feature_mean'], expected_mean, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(weights['feature_std'], expected_std.clamp_min(1e-3), atol=1e-5, rtol=1e-3)
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    ppo_events = [event for event in events if event['event'] == 'ppo']
    assert [event['feature_count'] for event in ppo_events] == [2 * (index+1) for index in range(len(calls))]


def test_fresh_mlp_outputs_ignore_statistics_but_trained_heads_do_not():
    """The CLI's pre-update seeding is only log-prob safe while the output heads are still zero."""
    generator = torch.Generator().manual_seed(4)
    features, previous = torch.randn(6, 768, generator=generator), torch.randn(6, 768, generator=generator)
    model = ActorCritic(vectors(), architecture='mlp')
    before = model(features, previous_features=previous)
    model.update_feature_statistics(torch.randn(8, 768, generator=generator))
    after = model(features, previous_features=previous)
    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])

    trained = ActorCritic(vectors(), architecture='mlp')
    with torch.no_grad():
        trained.nonlinear_actor[-1].weight.normal_(generator=generator)
    before = trained(features, previous_features=previous)[0]
    trained.update_feature_statistics(torch.randn(8, 768, generator=generator))
    assert not torch.allclose(trained(features, previous_features=previous)[0], before)


def test_resumed_trained_identity_checkpoint_keeps_acting_statistics_through_first_update(monkeypatch, tmp_path):
    # Regression: a migrated count of 0 must not trigger the fresh-run pre-update seeding.
    from mario_embedding_agent import cli
    install_fake_cli(monkeypatch, tmp_path, cli, [])
    old = ActorCritic(torch.ones(18, 768), architecture='mlp')
    with torch.no_grad():
        old.nonlinear_actor[-1].weight.normal_(generator=torch.Generator().manual_seed(5))
    weights = old.state_dict()
    weights.pop('feature_count')  # Checkpoint predates running statistics.
    optimizer = torch.optim.AdamW(old.parameters(), lr=3e-4)
    payload = {'config': {'image_tokens': 70, 'seed': 0, 'environment': 'SuperMarioBros-full-pixel-v1',
                          'learning_rate': 3e-4, 'policy_architecture': 'mlp',
                          'feature_standardization': 'identity'},
               'policy': weights, 'optimizer': optimizer.state_dict(),
               'training_state': {'frames': 10, 'decisions': 2, 'updates': 1, 'deaths': 0, 'clears': 0,
                                  'episode_seed': 0, 'episode_commands': [], 'stagnant_decisions': 0,
                                  'recovery_active': False, 'recovery_action_index': 0,
                                  'recovery_progress_window': [], 'visual_position_px': 0.,
                                  'visual_frontier_px': 0., 'torch_rng_state': torch.get_rng_state()}}
    at_update = []
    def fake_update(policy, optimizer, rollout, bootstrap, **kwargs):
        at_update.append((float(policy.feature_count), policy.feature_mean.clone(), policy.feature_std.clone()))
        return {}
    monkeypatch.setattr(cli, 'update', fake_update)
    monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync', '--fps', '0',
                                    '--total-frames', '12', '--rollout', '2', '--resume', 'old.pt'])
    cli.main()
    count, mean, std = at_update[0]
    # The rollout was collected with identity statistics, so PPO must evaluate it with them.
    assert count == 0.
    assert torch.all(mean == 0) and torch.all(std == 1)
    # Statistics are still merged after the update, so later rollouts are standardized.
    assert [entry[0] for entry in at_update[1:]] == [2. * index for index in range(1, len(at_update))]
