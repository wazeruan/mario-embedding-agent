import numpy as np
import pytest

from mario_embedding_agent.recovery import PipeExitRecovery
from mario_embedding_agent.rewards import GapDetector


def underground(x=232, feet=176, big=False, mario=True, tint=0):
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[208:] = (0, 168, 168)
    frame[48:208, 240:] = (0, 168, 0)
    frame[176:208, 208:240] = (0, 168, 0)
    if mario:
        height = 32 if big else 16
        frame[feet-height:feet, x-4:x+4] = np.clip(GapDetector.RED.astype(int)+tint, 0, 255)
        frame[feet-4:feet, x-2:x+2] = np.clip(GapDetector.SKIN.astype(int)+tint, 0, 255)
    return frame


@pytest.mark.parametrize('big', [False, True])
@pytest.mark.parametrize('tint', [-2, 0, 2])
def test_pipe_rgb_geometry_retreat_land_then_enter(big, tint):
    controller = PipeExitRecovery()
    on_pipe = underground(big=big, tint=tint)
    geometry = controller.geometry(on_pipe)
    assert geometry[:3] == (208, 176, 208)
    assert geometry[-1][1] == 176
    assert controller.choose(on_pipe) == ('left', 16, 'pipe_retreat')
    off_edge_falling = underground(x=190, feet=180, big=big, tint=tint)
    assert controller.choose(off_edge_falling) == ('idle', 8, 'pipe_land')
    grounded = underground(x=190, feet=208, big=big, tint=tint)
    assert controller.choose(grounded) == ('right', 24, 'pipe_enter')
    assert PipeExitRecovery().choose(grounded) == controller.choose(grounded)
    assert controller.choose(on_pipe) == ('left', 16, 'pipe_retreat')  # No stale phase memory.


@pytest.mark.parametrize('kind', ['overworld', 'no_mario', 'missing_low_pipe', 'missing_wall', 'far_airborne', 'ambiguous_sprite'])
def test_pipe_rgb_negatives_do_not_override(kind):
    frame = underground(mario=kind != 'no_mario')
    if kind == 'overworld':
        black = np.all(frame == 0, axis=2)
        frame[black] = GapDetector.SKY
    elif kind == 'missing_low_pipe':
        frame[176:208, 192:240] = 0
    elif kind == 'missing_wall':
        frame[48:208, 240:] = 0
    elif kind == 'far_airborne':
        frame = underground(x=100, feet=120)
    elif kind == 'ambiguous_sprite':
        frame[100:108, 64:72] = GapDetector.RED
    assert PipeExitRecovery().choose(frame) is None


@pytest.mark.parametrize(('frame', 'selected', 'phase', 'expected_duration'),
                         [(underground(), 'left', 'pipe_retreat', 16),
                          (underground(x=190, feet=180), 'idle', 'pipe_land', 8),
                          (underground(x=190, feet=208), 'right', 'pipe_enter', 24),
                          (underground(), 'right', 'left', 4)])
def test_cli_pipe_override_commands_match_selected_action_and_duration(monkeypatch, tmp_path, frame, selected, phase, expected_duration):
    import json
    import sys
    import torch
    from mario_embedding_agent import cli
    from mario_embedding_agent.actions import ACTIONS

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)
    class Env:
        def reset(self, seed):
            return frame.copy(), {}
        def close(self):
            pass
    commands, decisions = [], []
    def step(env, command):
        commands.append(command)
        return frame.copy(), False
    def act(self, features, sample=True, action_bias=None, previous_features=None):
        decisions.append(action_bias)
        if len(decisions) <= 4:
            assert action_bias is None
            index = 0
        else:
            assert action_bias is not None
            index = next(i for i, action in enumerate(ACTIONS) if action.name == selected)
        return index, 0., 0., [1.] + [0.] * 17
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', step)
    monkeypatch.setattr(cli.ActorCritic, 'act', act)
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync',
                                    '--total-frames', str(17+expected_duration), '--rollout', '2'])
    cli.main()
    wanted = next(action for action in ACTIONS if action.name == selected).commands()[0]
    assert commands[16:16+expected_duration] == [wanted]*expected_duration
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    result = [event for event in events if event['event'] == 'decision_result'][4]
    assert result['recovery_action'] == phase
    assert result['executed_frames'] == expected_duration
