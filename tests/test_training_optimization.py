"""Regressions for visual frontier rewards and duration-aware guarded PPO."""
import copy

import pytest
import torch

from mario_embedding_agent.actions import ACTIONS
from mario_embedding_agent.policy import ActorCritic
from mario_embedding_agent.ppo import Transition, gae, update
from mario_embedding_agent.rewards import ProgressFrontier, reward


def test_frontier_does_not_pay_for_retreat_or_retrace_and_resets_per_episode():
    frontier = ProgressFrontier()
    awards = []
    for displacement in [32., -16., 16., -32., 40.]:
        frontier.observe(displacement)
        awards.append(frontier.claim())
    assert awards == [32., 0., 0., 0., 8.]
    assert sum(reward(False, False, px, pixels_per_reward=32., deadband_px=0.) for px in awards) == 1.25
    assert frontier.position == frontier.frontier == 40.
    frontier.observe(None)
    assert frontier.claim() == 0.
    frontier.reset()
    assert frontier.position == frontier.frontier == 0.
    assert reward(False, True, 320., 1., clear_reward=50.) == 50.
    assert reward(True, True, 320., 1., clear_reward=50.) == -50.


def test_duration_discount_and_trace_use_actual_emulator_interval():
    features = torch.ones(768)
    transitions = [Transition(features, 0, 0., .2, .1, False, duration_frames=40),
                   Transition(features, 0, 0., .3, 1., True, duration_frames=4)]
    advantages, returns = gae(transitions, 100., gamma=.9, gae_lambda=.8)
    expected_last = 1. - .3
    expected_first = .1 + .9**2*.3 - .2 + .9**2*.8**2*expected_last
    torch.testing.assert_close(advantages, torch.tensor([expected_first, expected_last]))
    assert returns[-1] == 1.
    assert Transition(features, 0, 0., 0., 0., False).duration_frames == 20
    transitions[0].duration_frames = 0
    with pytest.raises(ValueError, match='positive'):
        gae(transitions, 0.)


def test_assisted_batch_updates_critic_but_not_actor_or_its_adam_momentum():
    torch.manual_seed(92)
    policy = ActorCritic(torch.randn(len(ACTIONS), 768))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.001, weight_decay=.1)
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    actor_parameters = [parameter for module in (policy.actor, policy.temporal_actor)
                        for parameter in module.parameters()]
    previous_weights = [parameter.detach().clone() for parameter in actor_parameters]
    previous_moments = [copy.deepcopy(optimizer.state[parameter]) for parameter in actor_parameters]
    previous_critic = policy.critic[-1].weight.detach().clone()
    bias = torch.zeros(len(ACTIONS)); bias[5] = 8.
    transitions = []
    for i in range(8):
        feature = torch.randn(768)
        action, log_prob, value, _ = policy.act(feature, action_bias=bias)
        transitions.append(Transition(feature, action, log_prob, value, 1., i == 7, bias))
    metrics = update(policy, optimizer, transitions, 0., epochs=2, minibatch_size=8)
    assert metrics['actor_samples'] == 0 and metrics['recovery_samples'] == 8
    assert metrics['optimizer_steps'] == 2
    assert not torch.equal(previous_critic, policy.critic[-1].weight)
    for parameter, weight, moments in zip(actor_parameters, previous_weights, previous_moments):
        torch.testing.assert_close(parameter, weight, rtol=0., atol=0.)
        for key, value in moments.items():
            torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0., atol=0.)


def test_kl_guard_rejects_stale_policy_before_any_optimizer_step():
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    feature = torch.ones(768)
    action, log_prob, value, _ = policy.act(feature, sample=False)
    transitions = [Transition(feature, action, log_prob, value, 1., True) for _ in range(8)]
    with torch.no_grad():
        policy.actor.bias[action] += 8.
    before = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    metrics = update(policy, optimizer, transitions, 0., epochs=4, minibatch_size=8, target_kl=.02)
    assert metrics['stopped_early'] and metrics['optimizer_steps'] == 0
    assert optimizer.state == {}
    for key, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0., atol=0.)


def test_realtime_releases_macro_buttons_while_waiting_and_records_executed_duration(monkeypatch, tmp_path):
    import json
    import sys
    import numpy as np
    from mario_embedding_agent import cli

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    class Future:
        def __init__(self, value, delay):
            self.value, self.delay, self.polls = value, delay, 0
        def done(self):
            self.polls += 1
            return self.polls > self.delay
        def result(self):
            return self.value

    class Executor:
        def __init__(self, **kwargs):
            self.calls = 0
        def submit(self, function, frame):
            self.calls += 1
            return Future(function(frame), 4 if self.calls == 2 else 0)
        def shutdown(self, **kwargs):
            pass

    commands, transitions = [], []
    def step(env, command):
        commands.append(command)
        return np.zeros((240, 256, 3), dtype=np.uint8), False
    def ppo(policy, optimizer, rollout, bootstrap, **kwargs):
        transitions.extend(copy.deepcopy(rollout))
        return {}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', step)
    monkeypatch.setattr(cli, 'ThreadPoolExecutor', Executor)
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (1, 0., 0., [0., 1.] + [0.] * 16))
    monkeypatch.setattr(cli, 'update', ppo)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'realtime',
                                    '--total-frames', '13', '--rollout', '2'])
    cli.main()
    assert commands[:4] == [('right',)] * 4
    assert commands[4:8] == [()] * 4
    assert commands[8:12] == [('right',)] * 4
    assert [transition.duration_frames for transition in transitions] == [8, 4]
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    second_input = [event for event in events if event['event'] == 'decision'][1]
    assert second_input['observation_age_frames'] == 4
    second_result = [event for event in events if event['event'] == 'decision_result'][1]
    assert second_result['executed_frames'] == 4  # Input age is excluded from PPO interval.


def test_mixed_batch_actor_metrics_exclude_assisted_samples_while_value_uses_all():
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.)
    feature = torch.ones(768)
    bias = torch.zeros(len(ACTIONS)); bias[5] = 8.
    transitions = []
    for amount, assistance in [(.1, None), (.3, None), (100., bias), (200., bias)]:
        action, log_prob, value, probabilities = policy.act(feature, action_bias=assistance)
        transitions.append(Transition(feature, action, log_prob, value, amount, True, assistance))
    _, _, _, native_probs = policy.act(feature)
    native_entropy = float(torch.distributions.Categorical(probs=torch.tensor(native_probs)).entropy())
    metrics = update(policy, optimizer, transitions, 0., epochs=1, minibatch_size=4)
    assert metrics['actor_samples'] == metrics['recovery_samples'] == 2
    assert metrics['policy_loss'] == pytest.approx(0., abs=1e-6)
    assert metrics['entropy'] == pytest.approx(native_entropy, abs=1e-6)
    assert metrics['value_loss'] == pytest.approx(.5*(.1**2+.3**2+100.**2+200.**2)/4, abs=.001)


def test_frontier_retains_within_action_excursion_then_never_repays_retrace():
    frontier = ProgressFrontier()
    frontier.observe(32.)
    frontier.observe(-32.)
    assert frontier.position == 0. and frontier.frontier == 32.
    assert frontier.claim() == 32.  # The endpoint alone would miss this exploration.
    assert frontier.claim() == 0.
    frontier.observe(32.)
    assert frontier.claim() == 0.  # Returning to the already explored peak is free.
    frontier.observe(8.)
    frontier.observe(-8.)
    assert frontier.claim() == 8.
    frontier.reset()
    assert frontier.position == frontier.frontier == frontier.pending == 0.


def test_restored_frontier_discards_partial_reward_but_keeps_explored_peak():
    original = ProgressFrontier()
    original.observe(40.)
    original.observe(-24.)
    restored = ProgressFrontier(original.position, original.frontier)
    assert original.pending == 40.
    assert restored.pending == 0.
    assert restored.claim() == 0.  # Resume cannot credit the interrupted action.
    restored.observe(24.)
    assert restored.claim() == 0.  # It cannot repay retracing explored ground either.
    restored.observe(8.)
    assert restored.claim() == 8.


def test_initial_inference_wait_progress_is_explored_but_not_paid_to_first_action(monkeypatch, tmp_path):
    import json
    import sys
    import numpy as np
    from mario_embedding_agent import cli

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    class Future:
        def __init__(self, value, delay):
            self.value, self.delay, self.polls = value, delay, 0
        def done(self):
            self.polls += 1
            return self.polls > self.delay
        def result(self):
            return self.value

    class Executor:
        def __init__(self, **kwargs):
            self.calls = 0
        def submit(self, function, frame):
            self.calls += 1
            return Future(function(frame), 3 if self.calls == 1 else 0)
        def shutdown(self, **kwargs):
            pass

    class Motion:
        def __init__(self):
            self.sequence = iter([1., 1., 1., 1., 1., 1., 1., -4., 4., 0., 0., 0.])
        def reset(self, frame):
            pass
        def observe(self, frame):
            return next(self.sequence)

    commands, transitions = [], []
    def step(env, command):
        commands.append(command)
        return np.zeros((240, 256, 3), dtype=np.uint8), False
    def ppo(policy, optimizer, rollout, bootstrap, **kwargs):
        transitions.extend(copy.deepcopy(rollout))
        return {}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', step)
    monkeypatch.setattr(cli, 'ThreadPoolExecutor', Executor)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (1, 0., 0., [0., 1.] + [0.] * 16))
    monkeypatch.setattr(cli, 'update', ppo)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'realtime',
                                    '--total-frames', '12', '--rollout', '2'])
    cli.main()
    assert commands[:3] == [()] * 3
    assert commands[3:7] == [('right',)] * 4
    assert [item.duration_frames for item in transitions] == [4, 4]
    assert [item.reward for item in transitions] == pytest.approx([4./32./15., 0.])
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    first_decision = next(event for event in events if event['event'] == 'decision')
    assert first_decision['visual_position_px'] == first_decision['visual_frontier_px'] == 3.
    first, second = [event for event in events if event['event'] == 'decision_result']
    assert first['new_frontier_px'] == 4.
    assert first['displacement_px'] == 4.
    assert second['new_frontier_px'] == 0.
    assert second['visual_frontier_px'] == 7.  # Initial wait ground remains explored.
