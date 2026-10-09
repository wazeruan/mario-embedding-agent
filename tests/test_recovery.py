import copy
import json
import sys

import numpy as np
import pytest
import torch

from mario_embedding_agent.actions import ACTIONS
from mario_embedding_agent.policy import ActorCritic
from mario_embedding_agent.ppo import Transition, update


def test_recovery_bias_replayed_by_ppo_has_unit_initial_ratio():
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.)
    features = torch.ones(768)
    bias = torch.zeros(len(ACTIONS))
    bias[next(i for i, action in enumerate(ACTIONS) if action.name == 'down')] = 8.
    rollout = []
    for i in range(8):
        action, log_prob, value, probabilities = policy.act(features, action_bias=bias)
        rollout.append(Transition(features, action, log_prob, value, .1, i == 7, bias.clone()))
    assert probabilities[5] > .99
    metrics = update(policy, optimizer, rollout, 0., epochs=1, minibatch_size=8)
    assert metrics['approx_kl'] == pytest.approx(0., abs=1e-7)
    assert metrics['clip_fraction'] == 0.
    assert all(torch.equal(item.action_bias, bias) for item in rollout)


@pytest.mark.parametrize('scenario', ['stalled', 'isolated_progress', 'death', 'gap'])
def test_cli_four_stagnant_decisions_schedule_and_keep_recovery_until_escape(monkeypatch, tmp_path, scenario):
    from mario_embedding_agent import cli

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        frames = 0
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    class Motion:
        def reset(self, frame):
            pass
        def observe(self, frame):
            frame_number = frame[0, 0, 0]
            if scenario == 'isolated_progress' and 17 <= frame_number <= 20:
                return 7.
            return 0.

    class Gap:
        def observe(self, frame):
            return scenario == 'gap' and frame[0, 0, 0] == 18

    env = Env()
    biases, transitions, commands = [], [], []
    def step(env, command):
        env.frames += 1
        commands.append(command)
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        frame[0, 0, 0] = env.frames
        return frame, scenario == 'death' and env.frames == 18
    def act(self, features, sample=True, action_bias=None, previous_features=None):
        biases.append(None if action_bias is None else action_bias.clone())
        action = 0 if action_bias is None else int(torch.argmax(action_bias))
        probabilities = [0.] * len(ACTIONS)
        probabilities[action] = 1.
        return action, 0., 0., probabilities
    def ppo(policy, optimizer, rollout, bootstrap, **kwargs):
        transitions.extend(copy.deepcopy(rollout))
        return {}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'step', step)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli, 'GapDetector', Gap)
    monkeypatch.setattr(cli.ActorCritic, 'act', act)
    monkeypatch.setattr(cli, 'update', ppo)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--teacher', '--headless', '--mode', 'sync',
                                    '--rollout', '2', '--total-frames',
                                    '45' if scenario != 'death' else '25'])
    cli.main()
    assert biases[:4] == [None] * 4
    down = next(i for i, action in enumerate(ACTIONS) if action.name == 'down')
    assert biases[4][down] == 8. and torch.count_nonzero(biases[4]) == 1
    assert torch.equal(transitions[4].action_bias, biases[4])
    assert commands.count(('down',)) == (2 if scenario == 'death' else 24)
    if scenario in ('death', 'gap'):
        assert biases[5] is None
    else:
        left = next(i for i, action in enumerate(ACTIONS) if action.name == 'run_left')
        assert biases[5][left] == 8.


def test_recovery_exits_after_two_progress_events_with_positive_window_net(monkeypatch, tmp_path):
    from mario_embedding_agent import cli

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        frames = 0
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    class Motion:
        def reset(self, frame):
            pass
        def observe(self, frame):
            return 2. if 17 <= frame[0, 0, 0] <= 44 else 0.

    env = Env()
    biases = []
    def step(env, command):
        env.frames += 1
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        frame[0, 0, 0] = env.frames
        return frame, False
    def act(self, features, sample=True, action_bias=None, previous_features=None):
        biases.append(None if action_bias is None else action_bias.clone())
        action = 0 if action_bias is None else int(torch.argmax(action_bias))
        return action, 0., 0., [float(i == action) for i in range(len(ACTIONS))]

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'step', step)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli.ActorCritic, 'act', act)
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--teacher', '--headless', '--mode', 'sync',
                                    '--rollout', '2', '--total-frames', '80'])

    cli.main()

    assert all(bias is not None for bias in biases[4:8])
    assert biases[8] is None
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    escaped = next(event for event in events if event.get('event') == 'decision_result' and
                   event.get('recovery_action') == 'right_jump_short')
    assert escaped['recovery_evidence_events'] == 2
    assert escaped['recovery_window_net_progress_px'] == 56.
