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


@pytest.mark.parametrize(('death', 'cleared', 'displacement', 'bonus', 'expected'),
                         [(False, False, 0., 0., 0.),
                          (False, False, 8., 0., 1.),
                          (False, False, -8., 0., -1.),
                          (False, False, 8., 1., 2.),
                          (False, False, 0., 1., 1.),
                          (False, False, 1., 0., 0.),
                          (False, False, -1., 0., 0.),
                          (False, False, 80., 0., 5.),
                          (False, False, -80., 0., -5.),
                          (True, False, 8., 1., -50.),
                          (False, True, 8., 1., 1.),
                          (True, True, 8., 1., -50.)])
def test_signed_reward_deadband_clip_and_terminal_override(death, cleared, displacement, bonus, expected):
    assert reward(death, cleared, displacement, bonus) == expected


def test_signed_progress_does_not_reward_equal_right_left_oscillation():
    assert sum(reward(False, False, pixels) for pixels in [8., -8., 16., -16.]) == 0.
    assert reward(False, False, 0.) == 0.
    assert reward(True, False, 8., death_penalty=20.) == -20.


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


@pytest.mark.parametrize('legacy_gap_config', [False, True])
@pytest.mark.parametrize('current_recovery_config', [False, True])
@pytest.mark.parametrize('legacy_visual_replay', [False, True])
def test_cli_migrates_legacy_reward_config_preserving_weights_and_counters(monkeypatch, tmp_path, legacy_gap_config, current_recovery_config, legacy_visual_replay):
    import copy
    import json
    import sys
    import torch
    from mario_embedding_agent import cli

    old_policy = cli.ActorCritic(torch.ones(18, 768))
    with torch.no_grad():
        old_policy.actor.bias.fill_(.42)
    legacy_parameters = [parameter for name, parameter in old_policy.named_parameters()
                         if not name.startswith(('temporal_actor.', 'temporal_critic.'))]
    old_optimizer = torch.optim.AdamW(legacy_parameters, lr=3e-4)
    # Populate Adam moments for actor and critic, including nonzero value output.
    for parameter in legacy_parameters:
        parameter.grad = torch.ones_like(parameter)
    old_optimizer.step()
    old_optimizer.zero_grad(set_to_none=True)
    old_config = {'reward_scale': 10., 'image_tokens': 70, 'seed': 7,
                  'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
                  'death_penalty': 10., 'pixels_per_reward': 8.,
                  'progress_clip': 5., 'reward_version': 'signed-progress-v1'}
    if legacy_gap_config:
        old_config.update(gap_bonus=1., gap_detector='overworld-rgb-v1',
                          movement_reward='rightward', motion_detector='rgb')
    if current_recovery_config:
        old_config.update(stagnation_threshold=4, stagnation_deadband_px=1.,
                          recovery_actions=['down', 'run_left', 'run_right_jump_long', 'right_jump_short'],
                          recovery_logit_bias=8., recovery_down_hold_frames=24,
                          recovery_progress_px=8., recovery_progress_window=4,
                          recovery_progress_events_required=2, recovery_window_net_progress_px=8., pipe_recovery='rgb-side-entry-v1')
    legacy_policy = {name: value for name, value in old_policy.state_dict().items()
                     if not name.startswith(('temporal_actor.', 'temporal_critic.'))}
    payload = {'config': old_config, 'policy': legacy_policy,
               'optimizer': old_optimizer.state_dict(),
               'training_state': {'frames': 100, 'decisions': 20, 'updates': 5,
                                  'deaths': 2, 'clears': 0, 'episode_seed': 9,
                                  'episode_commands': [], 'stagnant_decisions': 3, 'torch_rng_state': torch.get_rng_state()}}

    class Encoder:
        device = 'cpu'
        def __init__(self, *args, **kwargs):
            pass
        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        def reset(self, seed):
            assert seed == 9
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    if legacy_visual_replay:
        payload['training_state']['episode_commands'] = [cli.COMMANDS.index(('right',))]*3
        sequence = iter([10., -5., 5., 2.])
        class ReplayMotion:
            def reset(self, frame):
                pass
            def observe(self, frame):
                return next(sequence)
        monkeypatch.setattr(cli, 'RightwardProgress', ReplayMotion)

    saved = []
    def save(path, policy, optimizer, config, state):
        saved.append((copy.deepcopy(config), copy.deepcopy(state),
                      copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())))
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', lambda *args: (np.zeros((240, 256, 3), dtype=np.uint8), False))
    monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli.checkpoint, 'save', save)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--headless', '--mode', 'sync',
                                    '--total-frames', '1', '--resume', 'legacy.pt'])
    cli.main()
    config, state, weights, optimizer = saved[-1]
    assert config['reward_scale'] == 15.
    assert config['encoder_backend'] == 'embeddinggemma2'
    assert config['policy_architecture'] == 'legacy'
    assert config['feature_standardization'] == 'identity'
    assert config['student_fingerprint'] is None
    assert config['death_penalty'] == 15.
    assert config['clear_reward'] == 50.
    assert config['ppo_epochs'] == 4
    assert config['ppo_minibatch_size'] == 32
    assert config['value_clip'] is None
    assert config['target_kl'] == .02
    assert config['discount_reference_frames'] == 20
    assert config['optimizer_weight_decay'] == 0.
    assert config['pixels_per_reward'] == 32.
    assert config['progress_clip'] == 5.
    assert config['reward_version'] == 'visual-frontier-v4'
    assert config['gap_bonus'] == 1.
    assert config['left_action_logit_bias'] == -3.
    assert config['entropy_coef'] == .01
    assert config['stagnation_threshold'] == 4
    assert config['stagnation_deadband_px'] == 1.
    assert config['recovery_logit_bias'] == 8.
    assert config['recovery_actions'] == ['down', 'run_left', 'run_right_jump_long', 'right_jump_short']
    assert config['recovery_down_hold_frames'] == 24
    assert config['recovery_progress_px'] == 8.
    assert config['recovery_progress_window'] == 4
    assert config['recovery_progress_events_required'] == 2
    assert config['recovery_window_net_progress_px'] == 8.
    assert config['pipe_recovery'] == 'rgb-side-entry-v1'
    assert 'movement_reward' not in config and 'motion_detector' not in config
    assert state['frames'] == 101 and state['updates'] == 5 and state['deaths'] == 2
    assert state['episode_seed'] == 9
    assert state['stagnant_decisions'] == 3
    assert state['visual_position_px'] == (12. if legacy_visual_replay else 0.)
    assert state['visual_frontier_px'] == (12. if legacy_visual_replay else 0.)
    assert state['recovery_active'] is False
    assert state['recovery_action_index'] == 0
    assert state['recovery_progress_window'] == []
    assert torch.equal(weights['actor.bias'], payload['policy']['actor.bias'])
    assert torch.count_nonzero(weights['temporal_actor.weight']) == 0
    assert torch.count_nonzero(weights['temporal_critic.3.weight']) == 0
    assert torch.count_nonzero(weights['temporal_critic.3.bias']) == 0
    final_prefix = f'critic.{len(old_policy.critic)-1}.'
    for name, tensor in payload['policy'].items():
        if name.startswith(final_prefix):
            assert torch.count_nonzero(weights[name]) == 0
            assert torch.count_nonzero(tensor) > 0
        else:
            torch.testing.assert_close(weights[name], tensor)
    old_ids = payload['optimizer']['param_groups'][0]['params']
    actor_ids = [index for index, (name, _) in zip(old_ids, old_policy.named_parameters())
                 if name.startswith('actor.')]
    assert optimizer['state'] == {}
    assert optimizer['param_groups'][0]['weight_decay'] == 0.
    old_optimizer_ids = payload['optimizer']['param_groups'][0]['params']
    restored_optimizer_ids = optimizer['param_groups'][0]['params']
    assert restored_optimizer_ids[:len(old_optimizer_ids)] == old_optimizer_ids
    assert len(restored_optimizer_ids) == len(old_optimizer_ids) + 7
    events = [json.loads(line) for line in (tmp_path/'runs/train.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'started'
    assert events[0]['visual_position_px'] == (10. if legacy_visual_replay else 0.)
    assert events[0]['visual_frontier_px'] == (10. if legacy_visual_replay else 0.)
    assert events[0]['reward_objective_migrated'] is True
    assert events[0]['policy_behavior_migrated'] is True
    assert events[0]['critic_reset_for_reward_migration'] is True
    assert events[0]['recovery_behavior_migrated'] is (not current_recovery_config)
    assert events[0]['temporal_context_migrated'] is True
    assert events[0]['stagnant_decisions'] == 3
