import copy
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


@pytest.mark.parametrize('reset_reason', [None, 'progress', 'death', 'gap'])
def test_cli_four_stagnant_decisions_schedule_and_reset_recovery(monkeypatch, tmp_path, reset_reason):
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
            return 2. if reset_reason == 'progress' and 17 <= frame[0, 0, 0] <= 20 else 0.

    class Gap:
        def observe(self, frame):
            return reset_reason == 'gap' and frame[0, 0, 0] == 18

    env = Env()
    biases, transitions = [], []
    def step(env, command):
        env.frames += 1
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        frame[0, 0, 0] = env.frames
        return frame, reset_reason == 'death' and env.frames == 18
    def act(self, features, sample=True, action_bias=None, previous_features=None):
        biases.append(None if action_bias is None else action_bias.clone())
        return 0, 0., 0., [1.] + [0.] * 17
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
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync',
                                    '--rollout', '2', '--total-frames', '29'])
    cli.main()
    assert biases[:4] == [None] * 4
    down = next(i for i, action in enumerate(ACTIONS) if action.name == 'down')
    assert biases[4][down] == 8. and torch.count_nonzero(biases[4]) == 1
    assert torch.equal(transitions[4].action_bias, biases[4])
    if reset_reason:
        assert biases[5] is None
    else:
        left = next(i for i, action in enumerate(ACTIONS) if action.name == 'run_left')
        assert biases[5][left] == 8.
