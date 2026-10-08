import numpy as np
import pytest

from mario_embedding_agent.rewards import RightwardProgress, reward
from test_gap_rewards import overworld


def test_pixel_screen_motion_and_stationary_frames():
    motion = RightwardProgress()
    motion.reset(overworld(70))
    assert motion.observe(overworld(70)) == 0.
    assert motion.observe(overworld(74)) == 4.
    assert motion.observe(overworld(69)) == -5.
    assert motion.observe(overworld(69)) == 0.


def test_scrolling_scenery_adds_world_motion_with_stationary_screen_sprite():
    rng = np.random.default_rng(123)
    old = overworld(138, feet=180)
    old[208:] = rng.integers(0, 256, size=(32, 256, 3), dtype=np.uint8)
    new = old.copy()
    new[208:, :-3] = old[208:, 3:]
    new[208:, -3:] = rng.integers(0, 256, size=(32, 3, 3), dtype=np.uint8)
    motion = RightwardProgress()
    motion.reset(old)
    assert motion.observe(new) == 3.
    assert motion.observe(old) == -3.


def test_missing_sprite_contributes_no_guessed_displacement():
    motion = RightwardProgress()
    motion.reset(overworld(mario=False))
    assert motion.observe(overworld(mario=False)) is None
    assert motion.observe(overworld(70)) is None
    assert motion.observe(overworld(70)) == 0.


@pytest.mark.parametrize(('death', 'cleared', 'moving', 'bonus', 'expected'),
                         [(False, False, False, 0., 0.),
                          (False, False, True, 0., 1.),
                          (False, False, True, 1., 2.),
                          (False, False, False, 1., 1.),
                          (True, False, True, 1., -1000.),
                          (False, True, True, 1., 1.),
                          (True, True, True, 1., -1000.)])
def test_reward_no_decision_cost_and_terminal_override(death, cleared, moving, bonus, expected):
    assert reward(death, cleared, moving, bonus) == expected


def test_real_pixel_only_stationary_right_left_and_camera_scroll():
    from mario_embedding_agent.environment import make_env, step
    env = make_env()
    motion = RightwardProgress()
    try:
        frame, _ = env.reset(seed=0)
        motion.reset(frame)
        totals = []
        for count, command in [(40, ()), (40, ('right',)), (40, ('left',))]:
            displacements = []
            for _ in range(count):
                frame, died = step(env, command)
                assert not died
                displacement = motion.observe(frame)
                if displacement is not None:
                    displacements.append(displacement)
            totals.append(sum(displacements))
        assert totals[0] == 0.
        assert totals[1] > 30.
        assert totals[2] < -10.  # Net left despite initial rightward inertia.
        frame, _ = env.reset(seed=0)
        motion.reset(frame)
        scroll_total = progress_total = 0.
        for i in range(650):
            previous = frame
            jumping = i >= 20 and (i-20) % 36 < 34
            frame, died = step(env, ('right', 'B', 'A') if jumping else ('right', 'B'))
            assert not died
            displacement = motion.observe(frame)
            if i > 350:
                scroll_total += motion._scene_scroll(previous, frame)
                progress_total += displacement or 0.
        assert scroll_total > 500.
        assert progress_total > 500.
    finally:
        env.close()


def test_real_vertical_jump_without_horizontal_input_does_not_earn_rightward_motion():
    from mario_embedding_agent.environment import make_env, step
    env = make_env()
    motion = RightwardProgress()
    try:
        frame, _ = env.reset(seed=0)
        motion.reset(frame)
        progress = []
        for _ in range(24):
            frame, died = step(env, ('A',))
            assert not died
            displacement = motion.observe(frame)
            if displacement is not None:
                progress.append(displacement)
        assert sum(progress) <= 1.0
    finally:
        env.close()
