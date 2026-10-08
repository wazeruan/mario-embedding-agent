import numpy as np
import pytest

from mario_embedding_agent.environment import step, COMMANDS, make_env
from mario_embedding_agent.rewards import StageDetector, reward


def label_frame(digit=1):
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    label = frame[24:32, 152:176]
    label[:6, :2] = 255
    label[:6, 16:18] = 255
    label[3:5, 9:15] = 255
    label[digit, 3] = 255
    return frame


def test_pixel_level_detector_requires_stable_new_label_and_no_duplicates():
    detector = StageDetector()
    assert not detector.observe(label_frame(1))
    assert not detector.observe(label_frame(2))
    assert not detector.observe(np.zeros((240, 256, 3), dtype=np.uint8))
    assert not detector.observe(label_frame(2))
    assert not detector.observe(label_frame(2))
    assert detector.observe(label_frame(2))
    assert not detector.observe(label_frame(2))
    assert not detector.observe(label_frame(2))


@pytest.mark.parametrize(('death', 'clear', 'expected'),
                         [(False, False, 0.), (True, False, -50.),
                          (False, True, 1.), (True, True, -50.)])
def test_terminal_reward_overrides_shaping(death, clear, expected):
    assert reward(death, clear) == expected


def test_step_discards_reward_info_truncated_and_copies_pixels():
    class Forbidden:
        def __bool__(self):
            raise AssertionError('Forbidden game metadata consumed')

    frame = label_frame()
    class Env:
        def step(self, action):
            assert action == COMMANDS.index(('right',))
            return frame, Forbidden(), False, Forbidden(), Forbidden()

    pixels, died = step(Env(), ('right',))
    assert not died
    np.testing.assert_equal(pixels, frame)
    assert not np.shares_memory(pixels, frame)


def test_real_environment_has_pixel_observations_and_only_death_flag():
    env = make_env()
    try:
        frame, info = env.reset(seed=0)
        assert info == {}
        assert frame.shape == (240, 256, 3)
        assert frame.dtype == np.uint8
        raw = env.step(COMMANDS.index(()))
        assert raw[1] == 0.
        assert isinstance(raw[2], bool)
        assert raw[3] is False
        assert raw[4] == {}
    finally:
        env.close()


def test_death_on_ppo_boundary_autosaves_reset_episode(monkeypatch, tmp_path):
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
        resets = 0
        frames = 0
        def reset(self, seed):
            self.resets += 1
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    env = Env()
    saved = []
    updates = []
    def fake_step(env, command):
        env.frames += 1
        return np.zeros((240, 256, 3), dtype=np.uint8), env.frames == 8
    def fake_update(policy, optimizer, rollout, bootstrap, *, entropy_coef, **kwargs):
        assert entropy_coef == .001
        assert kwargs == dict(epochs=4, minibatch_size=16, target_kl=.02, reference_frames=20)
        updates.append(copy.deepcopy(rollout))
        return {}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli, 'update', fake_update)
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    monkeypatch.setattr(cli.checkpoint, 'save', lambda path, policy, optimizer, config, state: saved.append(copy.deepcopy(state)))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync',
                                    '--total-frames', '8', '--rollout', '2'])
    cli.main()
    assert env.resets == 2
    assert len(updates) == 1
    assert len(updates[0]) == 2
    assert updates[0][-1].done
    assert updates[0][-1].reward == -1.  # Scaled death reward.
    assert len(saved) == 2  # Autosave followed by final save.
    assert saved[0]['frames'] == 8
    assert saved[0]['deaths'] == 1
    assert saved[0]['updates'] == 1
    assert saved[0]['episode_seed'] == 1
    assert saved[0]['episode_commands'] == []
