"""Training-only restart curriculum: emulator snapshots every 64px of new ground let a
failed episode resume >=192px before where it ended, without double frontier reward."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from mario_embedding_agent import cli
from mario_embedding_agent.environment import COMMANDS, make_env, step

TIMEOUT = 1800
# Cycling actions makes the controller-command prefix non-trivial (4-frame macros).
CYCLE = (0, 1, 5, 6)  # idle, right, down, up


class Encoder:
    device = 'cpu'
    def __init__(self, *args, **kwargs):
        pass
    def encode_frame(self, frame):
        return torch.ones(768)


class Env:
    """`steps` is the restorable emulator position since reset; `total` counts every step.

    `path` is the controller-command history of the emulator's current state, so a
    correct checkpoint satisfies episode_commands == path (deterministic replay)."""
    def __init__(self):
        self.steps = 0
        self.total = 0
        self.path = []
        self.resets = []
        self.dumps = []
        self.loads = []
        self.unwrapped = self
    def reset(self, seed):
        self.resets.append(seed)
        self.steps = 0
        self.path = []
        return np.zeros((240, 256, 3), dtype=np.uint8), {}
    def dump_state(self):
        self.dumps.append(self.total)
        return {'steps': self.steps, 'path': list(self.path)}
    def load_state(self, snapshot):
        self.loads.append(snapshot['steps'])
        self.steps = snapshot['steps']
        self.path = list(snapshot['path'])
    def close(self):
        pass


class TorchProxy:
    """Replaces cli.torch so the restart coin flip is scripted; all else is real torch."""
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0
    def rand(self, *args, **kwargs):
        value = self.values[min(self.calls, len(self.values)-1)]
        self.calls += 1
        return torch.tensor(value)
    def __getattr__(self, name):
        return getattr(torch, name)


def run(monkeypatch, tmp_path, argv, *, motion=lambda env: 1.0, died=lambda env: False,
        cleared=lambda env: False, rand=(0.0,), payload=None):
    env = Env()
    saved = []
    proxy = TorchProxy(rand)

    class Motion:
        def reset(self, frame):
            pass
        def observe(self, frame):
            return motion(env)

    class Stage:
        def observe(self, frame):
            return cleared(env)

    class NoGap:
        def observe(self, frame):
            return False

    def fake_step(environment, command):
        environment.steps += 1
        environment.total += 1
        environment.path.append(COMMANDS.index(command))
        return np.zeros((240, 256, 3), dtype=np.uint8), died(environment)

    decisions = []
    def act(self, *args, **kwargs):
        decisions.append(None)
        return CYCLE[len(decisions) % len(CYCLE)], 0., 0., [1.] + [0.] * 17

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'StageDetector', Stage)
    monkeypatch.setattr(cli, 'GapDetector', NoGap)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.ActorCritic, 'act', act)
    monkeypatch.setattr(cli.checkpoint, 'save',
                        lambda path, policy, optimizer, config, state: saved.append((copy.deepcopy(config), copy.deepcopy(state))))
    if payload is not None:
        monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli, 'torch', proxy)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    argv = [*argv, '--teacher']
    if '--resume' in argv or argv[0] in ('play', 'evaluate'):
        (tmp_path/'checkpoints').mkdir(exist_ok=True)
        (tmp_path/'checkpoints/latest.pt').write_bytes(b'placeholder')  # load is faked
    monkeypatch.setattr(sys, 'argv', ['mario-agent', *argv])
    cli.main()
    command = argv[0]
    events = [json.loads(line) for line in (tmp_path/f'runs/{command}.jsonl').read_text().splitlines()]
    return env, events, saved, proxy


def train_argv(frames):
    return ['train', '--headless', '--mode', 'sync', '--total-frames', str(frames), '--rollout', '100000']


def at_total(*totals):
    return lambda env: env.total in totals


def kinds(events, kind):
    return [event for event in events if event['event'] == kind]


def test_death_restores_latest_snapshot_at_least_backoff_before_death(monkeypatch, tmp_path):
    # 1px of new ground per frame; snapshots at frontier 64..320; death at frontier 330.
    env, events, saved, proxy = run(monkeypatch, tmp_path, train_argv(340), died=at_total(330))
    assert env.dumps == [64, 128, 192, 256, 320]
    assert env.loads == [128]  # Latest with frontier <= 330-192 = 138 (64 also qualifies).
    assert env.resets == [0] and proxy.calls == 1
    restart, = kinds(events, 'restart')
    assert restart['from_frontier_px'] == 128. and restart['ended_frontier_px'] == 330.
    assert restart['restarts'] == 1 and restart['deaths'] == 1
    assert restart['visual_frontier_px'] == 128. and restart['visual_position_px'] == 128.
    config, state = saved[-1]
    assert (config['restart_probability'], config['restart_backoff_px'], config['restart_snapshot_px']) == (.75, 192., 64.)
    assert state['restarts'] == 1 and state['deaths'] == 1
    assert state['episode_seed'] == 0  # Same episode, no seed advance.
    # Restored command prefix + 10 new frames reconstructs the emulator state exactly.
    assert env.steps == 138
    assert state['episode_commands'] == env.path and len(set(env.path)) > 1
    assert state['visual_frontier_px'] == 138. and state['visual_position_px'] == 138.
    assert state['stall_frontier_px'] == 138. and state['stall_frames'] == 0


def test_first_decision_after_restore_pays_only_ground_beyond_snapshot(monkeypatch, tmp_path):
    _, events, _, _ = run(monkeypatch, tmp_path, train_argv(370), died=at_total(330))
    index = next(i for i, event in enumerate(events) if event['event'] == 'restart')
    after = [event for event in events[index:] if event['event'] == 'decision_result']
    # Decision at frame 330, finished at 334 after four frames of new ground 129..132.
    assert after[0]['frames'] == 334
    assert after[0]['new_frontier_px'] == 4. and after[0]['displacement_px'] == 4.
    assert after[0]['raw_reward'] == pytest.approx(4/32)
    assert all(event['new_frontier_px'] == 4. for event in after)
    assert sum(event['new_frontier_px'] for event in after) == 4*len(after) <= 370-330


@pytest.mark.parametrize(('death', 'rand'), [(200, 0.), (330, .75), (330, .9)])
def test_no_restore_without_qualifying_snapshot_or_when_coin_fails(monkeypatch, tmp_path, death, rand):
    env, events, saved, proxy = run(monkeypatch, tmp_path, train_argv(death+10),
                                    died=at_total(death), rand=(rand,))
    assert env.loads == [] and not kinds(events, 'restart')
    assert env.resets == [0, 1]
    # No qualifying snapshot (frontier <= 8) means no coin flip at all.
    assert proxy.calls == (0 if death == 200 else 1)
    state = saved[-1][1]
    assert state['restarts'] == 0 and state['deaths'] == 1 and state['episode_seed'] == 1
    assert state['episode_commands'] == env.path and len(env.path) == 10
    assert state['visual_frontier_px'] == 10.


def test_full_reset_discards_snapshots(monkeypatch, tmp_path):
    # Coin fails at the first death (frontier 330) -> full reset. In the new episode a
    # single terminal frame jumps 300px, so no new snapshot exists; stale ones would qualify.
    jump = 330+5
    env, events, saved, proxy = run(
        monkeypatch, tmp_path, train_argv(jump+3), died=at_total(330, jump),
        motion=lambda env: 300. if env.total == jump else 1., rand=(.9, 0.))
    assert env.loads == [] and env.resets == [0, 1, 2]
    assert proxy.calls == 0 + 1  # Second death had no candidate, so no flip.
    assert saved[-1][1]['restarts'] == 0 and saved[-1][1]['deaths'] == 2


def test_restore_drops_later_snapshots(monkeypatch, tmp_path):
    # Restore at 128 (death at 330). Two frames later a terminal frame jumps 300px to
    # frontier 429: candidates are <= 237; stale 192 must be gone, so 128 is reused.
    jump = 332
    env, events, saved, _ = run(
        monkeypatch, tmp_path, train_argv(jump+5), died=at_total(330, jump),
        motion=lambda env: 300. if env.total == jump else 1.)
    assert env.loads == [128, 128]
    assert [event['ended_frontier_px'] for event in kinds(events, 'restart')] == [330., 429.]
    state = saved[-1][1]
    assert state['restarts'] == 2 and state['deaths'] == 2 and env.resets == [0]
    assert state['episode_commands'] == env.path and env.steps == 133


def test_restore_snapshot_is_reusable_and_resnapshots_from_restored_frontier(monkeypatch, tmp_path):
    # Die at 330 -> restore 128; ground resumes, snapshots recaptured at 192, 256 and 320.
    second = 330 + (330-128)
    env, events, saved, _ = run(monkeypatch, tmp_path, train_argv(second+3), died=at_total(330, second))
    assert env.dumps == [64, 128, 192, 256, 320, 330+64, 330+128, 330+192]
    assert env.loads == [128, 128]
    assert saved[-1][1]['episode_commands'] == env.path


def test_stall_restores_and_restores_stall_reference(monkeypatch, tmp_path):
    # Ground stops after frontier 330 -> stall at 330+1800, restore at 128. With the stall
    # reference restored to 128, the next stall comes exactly TIMEOUT frames later.
    first = 330+TIMEOUT
    env, events, saved, _ = run(monkeypatch, tmp_path, train_argv(first+TIMEOUT+1),
                                motion=lambda env: 1. if env.total <= 330 else 0.)
    assert [event['frames'] for event in kinds(events, 'stall')] == [first, first+TIMEOUT]
    assert env.loads == [128]  # Second stall ended at 128: nothing 192px behind.
    restart, = kinds(events, 'restart')
    assert restart['stalls'] == 1 and restart['ended_frontier_px'] == 330.
    assert env.resets == [0, 1]
    assert saved[-1][1]['restarts'] == 1 and saved[-1][1]['stalls'] == 2


def test_level_clear_discards_snapshots(monkeypatch, tmp_path):
    # Clear at frontier 200; post-clear frontier rises to 9 and a terminal frame jumps
    # 300px (ended 310). Pre-clear snapshot 64 would qualify if it survived the clear.
    jump = 210
    env, events, saved, proxy = run(
        monkeypatch, tmp_path, train_argv(jump+3), cleared=at_total(200), died=at_total(jump),
        motion=lambda env: 300. if env.total == jump else 1.)
    assert env.dumps == [64, 128, 192]
    assert env.loads == [] and proxy.calls == 0 and env.resets == [0, 1]
    state = saved[-1][1]
    assert state['clears'] == 1 and state['deaths'] == 1 and state['restarts'] == 0


def test_level_clear_restarts_snapshot_milestones_in_new_level(monkeypatch, tmp_path):
    # After a clear at 200 the next snapshot is at post-clear frontier 64 (total 264).
    death = 200+300
    env, events, saved, _ = run(monkeypatch, tmp_path, train_argv(death+2),
                                cleared=at_total(200), died=at_total(death))
    assert env.dumps == [64, 128, 192, 264, 328, 392, 456]
    assert env.loads == [264]  # Post-clear frontier 64 <= 300-192.
    assert saved[-1][1]['episode_commands'] == env.path


def legacy_payload(training_state):
    policy = cli.ActorCritic(torch.ones(18, 768))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    config = {'reward_scale': 15.0, 'image_tokens': 70, 'seed': 0,
              'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
              'gap_bonus': 1.0, 'gap_detector': 'overworld-rgb-v1',
              'death_penalty': 15.0, 'clear_reward': 50.0, 'pixels_per_reward': 32.0,
              'progress_clip': 5.0, 'reward_version': 'visual-frontier-v5', 'stall_timeout_frames': 1800,
              'policy_architecture': 'legacy'}
    return {'config': config, 'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
            'training_state': {'frames': 100, 'decisions': 20, 'updates': 5, 'deaths': 2, 'clears': 0,
                               'episode_seed': 3, 'episode_commands': [], 'gaps': 1,
                               'stalls': 0, 'stall_frames': 0, 'stall_frontier_px': 0.,
                               'visual_position_px': 0., 'visual_frontier_px': 0.,
                               'torch_rng_state': torch.get_rng_state(), **training_state}}


@pytest.mark.parametrize('command', ['evaluate', 'play'])
def test_evaluate_and_play_never_snapshot_or_restore(monkeypatch, tmp_path, command):
    argv = ([command, '--headless'] +
            (['--episodes', '1', '--seed', '7'] if command == 'evaluate' else
             ['--mode', 'sync', '--total-frames', '340']))
    env, events, saved, proxy = run(monkeypatch, tmp_path, argv, died=at_total(330),
                                    payload=legacy_payload({}))
    assert env.dumps == [] and env.loads == [] and proxy.calls == 0
    assert not kinds(events, 'restart')
    assert env.resets == ([7] if command == 'evaluate' else [3, 4])
    assert saved == []


def test_old_checkpoint_without_restart_keys_resumes_and_restarts(monkeypatch, tmp_path):
    payload = legacy_payload({})
    assert 'restart_probability' not in payload['config'] and 'restarts' not in payload['training_state']
    env, events, saved, _ = run(monkeypatch, tmp_path, train_argv(340) + ['--resume'],
                                died=at_total(330), payload=payload)
    started, = kinds(events, 'started')
    assert started['restarts'] == 0 and started['reward_objective_migrated'] is False
    config, state = saved[-1]
    assert (config['restart_probability'], config['restart_backoff_px'], config['restart_snapshot_px']) == (.75, 192., 64.)
    assert state['restarts'] == 1 and state['deaths'] == 3 and state['frames'] == 440
    assert env.loads == [128] and env.resets == [3] and state['episode_seed'] == 3
    assert state['episode_commands'] == env.path


# Real emulator: restoring a snapshot then acting must equal replaying the command
# prefix from reset, which is how a checkpoint taken after a restart is resumed.

def test_real_emulator_snapshot_restore_matches_command_replay():
    prefix = [('right',)]*60 + [('right', 'B')]*20 + [('right', 'A')]*20
    continuation = [('right',)]*200  # Walks into the first goomba.
    further = [()]*10 + [('left',)]*20 + [('right', 'B', 'A')]*25 + [('right',)]*35

    def play(env, commands):
        frames = []
        for command in commands:
            frame, died = step(env, command)
            frames.append(frame)
            if died:
                break
        return frames, died

    env = make_env()
    try:
        env.reset(seed=0)
        prefix_frames, died = play(env, prefix)
        assert not died and len(prefix_frames) == len(prefix)
        snapshot = env.unwrapped.dump_state()
        continued, died = play(env, continuation)
        assert died  # The emulator is done; load_state must make it steppable again.
        env.unwrapped.load_state(snapshot)
        restored, restored_died = play(env, further)
        assert len(restored) > 1
        # Same emulator state as the continuation's start, then a different branch.
        np.testing.assert_array_equal(restored[0], continued[0])  # Input acts a frame late.
        assert any(not np.array_equal(a, b) for a, b in zip(restored, continued))

        env.reset(seed=1)  # The seed does not alter emulation; the replay resets anyway.
        replay_prefix, died = play(env, prefix)
        assert not died
        np.testing.assert_array_equal(replay_prefix[-1], prefix_frames[-1])
        replayed, replayed_died = play(env, further)
        assert replayed_died == restored_died and len(replayed) == len(restored)
        for actual, expected in zip(replayed, restored):
            np.testing.assert_array_equal(actual, expected)
    finally:
        env.close()

    fresh = make_env()  # A separate process-level emulator, as on resume.
    try:
        fresh.reset(seed=0)
        frames, died = play(fresh, prefix + further[:len(restored)])
        assert died == restored_died
        np.testing.assert_array_equal(frames[-1], restored[-1])
    finally:
        fresh.close()
