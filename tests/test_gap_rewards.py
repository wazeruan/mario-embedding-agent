"""Pixel-only gap shaping: synthetic edge cases and real emulator replay."""
import numpy as np

from mario_embedding_agent.rewards import GapDetector


def overworld(x=70, feet=208, gap=(100, 132), mario=True):
    frame = np.empty((240, 256, 3), dtype=np.uint8)
    frame[:] = GapDetector.SKY
    frame[208:] = (228, 92, 16)
    if gap:
        frame[208:, gap[0]:gap[1]] = GapDetector.SKY
    if mario:
        frame[feet-16:feet, x-4:x+4] = GapDetector.RED
        frame[feet-4:feet, x-2:x+2] = GapDetector.SKIN
    return frame


def approach(detector):
    for _ in range(3):
        assert not detector.observe(overworld())


def test_gap_rewards_only_after_airborne_crossing_and_stable_landing():
    detector = GapDetector()
    approach(detector)
    assert not detector.observe(overworld(90, 180))
    assert not detector.observe(overworld(116, 178))
    assert not detector.observe(overworld(145))
    assert not detector.observe(overworld(145))
    assert detector.observe(overworld(145))
    for _ in range(20):
        assert not detector.observe(overworld(145))
    # Walking back and re-crossing the same visible hole cannot farm bonuses.
    assert not detector.observe(overworld(70))
    assert not detector.observe(overworld(116, 180))
    for _ in range(3):
        assert not detector.observe(overworld(145))


def test_standing_enemy_jump_and_walking_or_falling_into_gap_not_rewarded():
    detector = GapDetector()
    approach(detector)
    for frame in [overworld(70), overworld(70, 180), overworld(70),
                  overworld(110, 208), overworld(115, 214), overworld(125, 225)]:
        assert not detector.observe(frame)
    # No hole means no bonus for a perfectly normal jump or stomp.
    detector = GapDetector()
    for frame in [overworld(70, gap=None)] * 3 + [overworld(90, 180, gap=None),
                                               overworld(145, gap=None)] * 3:
        assert not detector.observe(frame)


def test_unsupported_palette_unbounded_hole_and_sprite_absence_fail_closed():
    for gap in [(0, 32), (220, 256), (100, 210)]:
        detector = GapDetector()
        for frame in [overworld(70, gap=gap)] * 3 + [overworld(115, 180, gap=gap)] + [overworld(220, gap=gap)] * 3:
            assert not detector.observe(frame)
    detector = GapDetector()
    for _ in range(5):
        assert not detector.observe(overworld(mario=False))
        assert not detector.observe(np.zeros((240, 256, 3), dtype=np.uint8))


def test_real_overworld_gap_crossing_from_controller_replay():
    # Actual RGB observations, deterministic controller sequence; no RAM/info/reward.
    from mario_embedding_agent.environment import make_env, step
    env = make_env()
    detector = GapDetector()
    observed_crossings = []
    try:
        frame, _ = env.reset(seed=0)
        detector.observe(frame)
        for frame_index in range(650):
            jumping = frame_index >= 20 and (frame_index-20) % 36 < 34
            frame, died = step(env, ('right', 'B', 'A') if jumping else ('right', 'B'))
            if detector.observe(frame):
                observed_crossings.append(frame_index)
            assert not died
        assert observed_crossings == [611]
    finally:
        env.close()


import pytest


@pytest.mark.parametrize(('cross_at', 'death_at', 'expected_rewards', 'expected_gaps', 'motion_value'),
                         [(2, None, [.1, 0.], 1, 0.),
                          (6, 8, [0., -1.], 1, 0.),
                          (2, 2, [-1., 0.], 0, 0.),
                          (2, None, [.13, .03], 1, .6),
                          (2, None, [.07, -.03], 1, -.6),
                          (6, 8, [.03, -1.], 1, .6)])
def test_cli_attributes_bonus_to_active_decision_and_death_overrides(
        monkeypatch, tmp_path, cross_at, death_at, expected_rewards, expected_gaps, motion_value):
    import copy
    import sys
    import torch
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

    class Detector:
        def observe(self, frame):
            return int(frame[0, 0, 0]) == cross_at

    class Motion:
        def reset(self, frame):
            pass
        def observe(self, frame):
            return motion_value

    env = Env()
    updates, saved = [], []
    def fake_step(env, command):
        env.frames += 1
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        frame[0, 0, 0] = env.frames
        return frame, env.frames == death_at
    def fake_update(policy, optimizer, transitions, bootstrap, *, entropy_coef):
        assert entropy_coef == .001
        updates.append(copy.deepcopy(transitions))
        return {}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'GapDetector', Detector)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli, 'update', fake_update)
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    monkeypatch.setattr(cli.checkpoint, 'save', lambda path, policy, optimizer, config, state: saved.append(copy.deepcopy(state)))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync',
                                    '--total-frames', '10', '--rollout', '2'])
    cli.main()
    assert [item.reward for item in updates[0]] == pytest.approx(expected_rewards)
    assert saved[-1]['gaps'] == expected_gaps
    if death_at is not None:
        assert any(item.done and item.reward == -1. for item in updates[0])
