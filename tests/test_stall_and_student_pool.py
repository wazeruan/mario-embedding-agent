"""Training stall termination (no new visual ground) and student pooling-grid persistence."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from mario_embedding_agent import cli, distill
from mario_embedding_agent.encoder import action_cache_metadata
from mario_embedding_agent.student import SmallEncoder, StudentEncoder, preprocess, save_student

TIMEOUT = 1800


class Encoder:
    device = 'cpu'
    def __init__(self, *args, **kwargs):
        pass
    def encode_frame(self, frame):
        return torch.ones(768)


class Env:
    """Counts every emulator step globally; frames carry no information."""
    def __init__(self):
        self.steps = 0
        self.resets = []
    def reset(self, seed):
        self.resets.append(seed)
        return np.zeros((240, 256, 3), dtype=np.uint8), {}
    def close(self):
        pass


def run(monkeypatch, tmp_path, argv, *, motion=None, death_at=(), clear_at=(), payload=None):
    """Drive cli.main with fake pixels. motion maps global step -> displacement px."""
    env = Env()
    motion = motion or {}
    transitions, saved = [], []

    class Motion:
        def reset(self, frame):
            pass
        def observe(self, frame):
            return motion.get(env.steps, 0.)

    class Stage:
        def observe(self, frame):
            return env.steps in clear_at

    class NoGap:
        def observe(self, frame):
            return False

    def fake_step(environment, command):
        environment.steps += 1
        return np.zeros((240, 256, 3), dtype=np.uint8), environment.steps in death_at

    real_transition = cli.Transition
    def record_transition(*args):
        item = real_transition(*args)
        transitions.append(item)
        return item

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'StageDetector', Stage)
    monkeypatch.setattr(cli, 'GapDetector', NoGap)
    monkeypatch.setattr(cli, 'RightwardProgress', Motion)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli, 'Transition', record_transition)
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    monkeypatch.setattr(cli.checkpoint, 'save',
                        lambda path, policy, optimizer, config, state: saved.append((copy.deepcopy(config), copy.deepcopy(state))))
    if payload is not None:
        monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', *argv])
    cli.main()
    command = argv[0]
    events = [json.loads(line) for line in (tmp_path/f'runs/{command}.jsonl').read_text().splitlines()]
    return env, events, transitions, saved


def train_argv(frames):
    # Huge rollout so PPO updates never interleave; the final save still records state.
    return ['train', '--headless', '--mode', 'sync', '--total-frames', str(frames), '--rollout', '100000']


def test_train_stalls_after_exactly_timeout_frames_without_new_ground(monkeypatch, tmp_path):
    env, events, transitions, saved = run(monkeypatch, tmp_path, train_argv(TIMEOUT+5))
    stalls = [event for event in events if event['event'] == 'stall']
    assert len(stalls) == 1
    assert stalls[0]['frames'] == TIMEOUT
    assert stalls[0]['raw_reward'] == -15.
    assert stalls[0]['stalls'] == 1 and stalls[0]['deaths'] == 0
    assert stalls[0]['stall_frames'] == 0  # Counter reset with the episode.
    assert (tmp_path/f'runs/stall-{TIMEOUT}.png').exists()
    assert not list((tmp_path/'runs').glob('death-*.png'))
    # Same reset path as death: next episode seed.
    assert env.resets == [0, 1]
    results = [event for event in events if event['event'] == 'decision_result']
    terminal = [event for event in results if event['outcome'] == 'stall']
    assert len(terminal) == 1 and terminal[0]['raw_reward'] == -15.
    assert not any(event['outcome'] == 'death' for event in results)
    done = [item for item in transitions if item.done]
    assert len(done) == 1 and done[0].reward == pytest.approx(-1.)  # -death_penalty/reward_scale
    # The frame before the timeout was not terminal.
    before = [event for event in events if event.get('frames') == TIMEOUT-1 and 'stall_frames' in event]
    assert all(event['stall_frames'] <= TIMEOUT-1 for event in before)
    config, state = saved[-1]
    assert config['stall_timeout_frames'] == TIMEOUT
    assert config['reward_version'] == 'visual-frontier-v5'
    assert state['stalls'] == 1 and state['deaths'] == 0
    assert state['stall_frames'] == 5 and state['stall_frontier_px'] == 0.
    assert state['episode_seed'] == 1
    assert len(state['episode_commands']) == 5
    # Recovery had engaged during the stall and is reset by the stall like a death.
    assert any(event['recovery_action'] for event in results if event['frames'] < TIMEOUT)
    assert stalls[0]['recovery_active'] is False and stalls[0]['stagnant_decisions'] == 0
    assert stalls[0]['recovery_progress_window'] == []


def test_train_does_not_stall_one_frame_before_timeout(monkeypatch, tmp_path):
    env, events, _, saved = run(monkeypatch, tmp_path, train_argv(TIMEOUT-1))
    assert not any(event['event'] == 'stall' for event in events)
    assert env.resets == [0]
    assert saved[-1][1]['stall_frames'] == TIMEOUT-1 and saved[-1][1]['stalls'] == 0


def test_new_frontier_resets_counter_but_retracing_old_ground_does_not(monkeypatch, tmp_path):
    # Step 500 retreats, 600 returns to (not beyond) the frontier, 1000 reaches new ground.
    motion = {500: -5., 600: 5., 1000: 1.}
    env, events, _, saved = run(monkeypatch, tmp_path, train_argv(1000+TIMEOUT+3), motion=motion)
    stalls = [event for event in events if event['event'] == 'stall']
    assert [event['frames'] for event in stalls] == [1000+TIMEOUT]
    assert env.resets == [0, 1]
    state = saved[-1][1]
    assert state['stalls'] == 1 and state['stall_frames'] == 3
    assert state['visual_frontier_px'] == 0.  # Frontier reset with the episode.


@pytest.mark.parametrize('terminal', ['death', 'clear'])
def test_death_or_clear_on_timeout_frame_takes_precedence(monkeypatch, tmp_path, terminal):
    kwargs = {'death_at': {TIMEOUT}} if terminal == 'death' else {'clear_at': {TIMEOUT}}
    env, events, transitions, saved = run(monkeypatch, tmp_path, train_argv(TIMEOUT+2), **kwargs)
    assert not any(event['event'] == 'stall' for event in events)
    assert not list((tmp_path/'runs').glob('stall-*.png'))
    assert (tmp_path/f'runs/{terminal}-{TIMEOUT}.png').exists()
    state = saved[-1][1]
    assert state['stalls'] == 0
    assert state['stall_frames'] == 2  # Counter reset by the terminal event.
    if terminal == 'death':
        assert state['deaths'] == 1 and env.resets == [0, 1]
        assert [item.reward for item in transitions if item.done] == pytest.approx([-1.])
    else:
        assert state['clears'] == 1 and state['deaths'] == 0 and env.resets == [0]
        assert [item.reward for item in transitions if item.done] == pytest.approx([50/15])


def legacy_payload(training_state):
    policy = cli.ActorCritic(torch.ones(18, 768))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    config = {'reward_scale': 15.0, 'image_tokens': 70, 'seed': 0,
              'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
              'gap_bonus': 1.0, 'gap_detector': 'overworld-rgb-v1',
              'death_penalty': 15.0, 'clear_reward': 50.0, 'pixels_per_reward': 32.0,
              'progress_clip': 5.0, 'reward_version': 'visual-frontier-v4',
              'policy_architecture': 'legacy'}
    return {'config': config, 'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
            'training_state': {'frames': 100, 'decisions': 20, 'updates': 5, 'deaths': 2, 'clears': 0,
                               'episode_seed': 3, 'episode_commands': [], 'gaps': 1,
                               'torch_rng_state': torch.get_rng_state(), **training_state}}


def test_evaluate_never_stall_terminates(monkeypatch, tmp_path):
    payload = legacy_payload({})
    env, events, transitions, saved = run(
        monkeypatch, tmp_path, ['evaluate', '--headless', '--episodes', '1', '--seed', '7', '--checkpoint', 'p.pt'],
        death_at={TIMEOUT+200}, payload=payload)
    assert not any(event['event'] == 'stall' for event in events)
    assert not list((tmp_path/'runs').glob('stall-*.png'))
    episode = next(event for event in events if event['event'] == 'evaluation_episode')
    assert episode['frames'] == TIMEOUT+200 and episode['raw_reward'] == -15.
    assert env.resets == [7]
    assert saved == [] and transitions == []


def test_legacy_checkpoint_without_stall_keys_resumes(monkeypatch, tmp_path):
    payload = legacy_payload({'visual_position_px': 40., 'visual_frontier_px': 50.})
    _, events, _, saved = run(monkeypatch, tmp_path, train_argv(3) + ['--resume', 'legacy.pt'], payload=payload)
    config, state = saved[-1]
    assert config['stall_timeout_frames'] == TIMEOUT
    assert config['reward_version'] == 'visual-frontier-v5'
    assert state['stalls'] == 0 and state['deaths'] == 2 and state['frames'] == 103
    # First frame adopts the restored frontier as the stall reference, then counts.
    assert state['stall_frontier_px'] == 50. and state['stall_frames'] == 2
    started = next(event for event in events if event['event'] == 'started')
    assert started['stalls'] == 0 and started['stall_frames'] == 0
    assert started['reward_objective_migrated'] is True


def test_resumed_stall_counter_continues_across_checkpoint(monkeypatch, tmp_path):
    payload = legacy_payload({'stalls': 4, 'stall_frames': TIMEOUT-5, 'stall_frontier_px': 0.,
                              'visual_position_px': 0., 'visual_frontier_px': 0.})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT)
    env, events, _, saved = run(monkeypatch, tmp_path, train_argv(7) + ['--resume', 'x.pt'], payload=payload)
    stalls = [event for event in events if event['event'] == 'stall']
    assert [event['frames'] for event in stalls] == [105]
    assert env.resets == [3, 4]
    state = saved[-1][1]
    assert state['stalls'] == 5 and state['deaths'] == 2 and state['stall_frames'] == 2


def test_student_pool_roundtrip_and_legacy_payload(tmp_path):
    torch.set_num_threads(1)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[100:140, 60:80] = (200, 50, 20)
    model = SmallEncoder(pool=8).eval()
    assert model.net[-3].in_features == 64*8*8
    with torch.no_grad():
        expected = model(preprocess(frame))[0]
    assert expected.shape == (768,)
    path = tmp_path/'student8.pt'
    save_student(model, path)
    assert torch.load(path, weights_only=True)['pool'] == 8
    student = StudentEncoder(path)
    assert student.model.pool == 8
    actual = student.encode_frame(frame)
    assert actual.shape == (768,)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    # Files written before 'pool' existed were always 4x4.
    legacy = SmallEncoder().eval()
    assert legacy.pool == 4 and legacy.net[-3].in_features == 1024
    legacy_path = tmp_path/'legacy.pt'
    torch.save({'format': 'mario-student-v1', 'metadata': action_cache_metadata(70),
                'weights': {k: v.detach().cpu() for k, v in legacy.state_dict().items()},
                'metrics': {}}, legacy_path)
    loaded = StudentEncoder(legacy_path)
    assert loaded.model.pool == 4
    with torch.no_grad():
        torch.testing.assert_close(loaded.encode_frame(frame), legacy(preprocess(frame))[0], rtol=0., atol=0.)
    # A legacy payload can't silently load into an 8x8 model.
    with pytest.raises(RuntimeError):
        SmallEncoder(pool=8).load_state_dict(torch.load(legacy_path, weights_only=True)['weights'])


def test_distill_train_pool_flag_reaches_saved_student(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(distill, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'train', '--pool', '8', '--data', str(tmp_path/'d.pt'),
                                      '--output', str(tmp_path/'s.pt')])
    distill.main()
    assert calls == [((tmp_path/'d.pt', tmp_path/'s.pt', 60, 0, 8), {})]
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'train'])
    distill.main()
    assert calls[-1][0][-1] == 8  # 8x8 is the default for newly trained students.


def test_distill_train_with_pool_8_saves_loadable_student(tmp_path, monkeypatch):
    from torch.nn import functional as F
    count = 64
    frames = torch.zeros(count, 240, 256, 3, dtype=torch.uint8)
    frames[:, 0, 0, 0] = torch.arange(count, dtype=torch.uint8)
    targets = F.normalize(torch.randn(count, 768, generator=torch.Generator().manual_seed(1)), dim=1)
    groups = torch.tensor([(i % 4, i // 16) for i in range(count)])
    data_path, student_path = tmp_path/'data.pt', tmp_path/'student.pt'
    torch.save(dict(frames=frames, targets=targets, groups=groups, metadata=action_cache_metadata(70)), data_path)
    distill.train(data_path, student_path, epochs=1, pool=8)
    assert torch.load(student_path, weights_only=True)['pool'] == 8
    student = StudentEncoder(student_path)
    assert student.model.pool == 8
    assert student.encode_frame(np.zeros((240, 256, 3), dtype=np.uint8)).shape == (768,)


@pytest.mark.parametrize('case', ['fresh_custom', 'fresh_default', 'resume'])
def test_student_mlp_calibrates_from_calibration_frames_only_when_fresh(monkeypatch, tmp_path, case):
    import mario_embedding_agent.student as student_module
    from mario_embedding_agent.policy import ActorCritic
    path = tmp_path/'student.pt'
    save_student(SmallEncoder(), path)
    cache = tmp_path/'models/action-embeddings.pt'
    cache.parent.mkdir()
    torch.save(dict(metadata=action_cache_metadata(70), embeddings=torch.ones(18, 768)), cache)
    mean, std = torch.full((768,), .01), torch.full((768,), .02)
    calls = []
    def calibrate(encoder, frames_path):
        assert isinstance(encoder, StudentEncoder)
        calls.append(frames_path)
        return mean, std
    monkeypatch.setattr(student_module, 'calibrate_policy_features', calibrate)
    policy = ActorCritic(torch.ones(18, 768), architecture='mlp', feature_mean=mean, feature_std=std)
    payload = {'config': {'reward_scale': 15., 'image_tokens': 70, 'seed': 0,
                          'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
                          'encoder_backend': 'student', 'policy_architecture': 'mlp',
                          'student_fingerprint': StudentEncoder(path).fingerprint},
               'policy': policy.state_dict(),
               'optimizer': torch.optim.AdamW(policy.parameters(), lr=3e-4).state_dict(),
               'training_state': {'episode_seed': 0, 'episode_commands': [],
                                  'torch_rng_state': torch.get_rng_state()}}

    class Env:
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass
    saved = []
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Teacher initialized'))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', lambda *args: (np.zeros((240, 256, 3), dtype=np.uint8), False))
    monkeypatch.setattr(cli.checkpoint, 'load', lambda p: payload)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda p, policy, optimizer, config, state: saved.append(dict(config)))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    custom = tmp_path/'elsewhere/calibration.pt'
    argv = ['mario-agent', 'train', '--student-encoder', str(path), '--headless', '--mode', 'sync',
            '--total-frames', '1', '--policy-architecture', 'mlp']
    if case != 'fresh_default':
        argv += ['--calibration-frames', str(custom)]
    if case == 'resume':
        argv += ['--resume', 'resume.pt']
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    expected = {'fresh_custom': [custom], 'fresh_default': [tmp_path/'models/distillation/frames.pt'],
                'resume': []}[case]
    assert calls == expected
    assert saved[-1]['encoder_backend'] == 'student' and saved[-1]['policy_architecture'] == 'mlp'
