"""Training stall termination, IMPALA student persistence, and CLI checkpoint/default behaviour."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from mario_embedding_agent import cli, distill
from mario_embedding_agent.encoder import action_cache_metadata
from mario_embedding_agent.student import StudentCNN, StudentEncoder, preprocess, save_student

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


def run(monkeypatch, tmp_path, argv, *, motion=None, death_at=(), clear_at=(), payload=None, update=None):
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
    monkeypatch.setattr(cli, 'update', update or (lambda *args, **kwargs: {}))
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    env.save_paths = []
    def save(path, policy, optimizer, config, state):
        env.save_paths.append(path)
        saved.append((copy.deepcopy(config), copy.deepcopy(state)))
    monkeypatch.setattr(cli.checkpoint, 'save', save)
    if payload is not None:
        monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    if '--student-encoder' not in argv:
        argv = [*argv, '--teacher']  # FrozenEncoder above is the fake.
    if '--resume' in argv or argv[0] in ('play', 'evaluate'):
        (tmp_path/'checkpoints').mkdir(exist_ok=True)
        (tmp_path/'checkpoints/latest.pt').write_bytes(b'placeholder')  # load is faked
    monkeypatch.setattr(sys, 'argv', ['mario-agent', *argv])
    cli.main()
    command = argv[0]
    events = [json.loads(line) for line in (tmp_path/f'runs/{command}.jsonl').read_text().splitlines()]
    return env, events, transitions, saved


def train_argv(frames):
    # Huge rollout so PPO updates never interleave; the final save still records state.
    return ['train', '--total-frames', str(frames), '--rollout', '100000']


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
    (tmp_path/'p.pt').write_bytes(b'placeholder')
    env, events, transitions, saved = run(
        monkeypatch, tmp_path, ['evaluate', '--episodes', '1', '--seed', '7', '--checkpoint', str(tmp_path/'p.pt')],
        death_at={TIMEOUT+200}, payload=payload)
    assert not any(event['event'] == 'stall' for event in events)
    assert not list((tmp_path/'runs').glob('stall-*.png'))
    episode = next(event for event in events if event['event'] == 'evaluation_episode')
    assert episode['frames'] == TIMEOUT+200 and episode['raw_reward'] == -15.
    assert env.resets == [7]
    assert saved == [] and transitions == []


def test_legacy_checkpoint_without_stall_keys_resumes(monkeypatch, tmp_path):
    payload = legacy_payload({'visual_position_px': 40., 'visual_frontier_px': 50.})
    _, events, _, saved = run(monkeypatch, tmp_path, train_argv(3) + ['--resume'], payload=payload)
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
    env, events, _, saved = run(monkeypatch, tmp_path, train_argv(7) + ['--resume'], payload=payload)
    stalls = [event for event in events if event['event'] == 'stall']
    assert [event['frames'] for event in stalls] == [105]
    assert env.resets == [3, 4]
    state = saved[-1][1]
    assert state['stalls'] == 5 and state['deaths'] == 2 and state['stall_frames'] == 2


def test_distill_cli_pool_and_width_reach_train(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(distill, 'ROOT', tmp_path)  # Default --output must not exist yet.
    monkeypatch.setattr(distill, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'train', '--data', str(tmp_path/'d.pt'),
                                      '--output', str(tmp_path/'s.pt')])
    distill.main()
    assert calls == [((tmp_path/'d.pt', tmp_path/'s.pt', 60, 0, 8, 1), {})]
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'train', '--width', '1.5', '--pool', '4'])
    distill.main()
    assert calls[-1][0][-2:] == (4, 1.5)
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'train', '--arch', 'impala'])
    with pytest.raises(SystemExit) as exit_info:  # --arch no longer exists.
        distill.main()
    assert exit_info.value.code == 2 and len(calls) == 2


def test_distill_train_saves_loadable_impala_student(tmp_path):
    from torch.nn import functional as F
    count = 64
    frames = torch.zeros(count, 240, 256, 3, dtype=torch.uint8)
    frames[:, 0, 0, 0] = torch.arange(count, dtype=torch.uint8)
    targets = F.normalize(torch.randn(count, 768, generator=torch.Generator().manual_seed(1)), dim=1)
    groups = torch.tensor([(i % 4, i // 16) for i in range(count)])
    data_path, student_path = tmp_path/'data.pt', tmp_path/'student.pt'
    torch.save(dict(frames=frames, targets=targets, groups=groups, metadata=action_cache_metadata(70)), data_path)
    distill.train(data_path, student_path, epochs=1, pool=4, width=.5)
    payload = torch.load(student_path, weights_only=True)
    assert (payload['arch'], payload['pool'], payload['width']) == ('impala', 4, .5)
    student = StudentEncoder(student_path)
    assert (student.model.pool, student.model.width) == (4, .5)
    assert student.encode_frame(np.zeros((240, 256, 3), dtype=np.uint8)).shape == (768,)


@pytest.mark.parametrize(('command', 'flag'), [('collect', '--data'), ('train', '--output')])
def test_distill_cli_refuses_to_overwrite_existing_output(tmp_path, monkeypatch, capsys, command, flag):
    target = tmp_path/'existing.pt'
    target.write_bytes(b'keep me')
    monkeypatch.setattr(distill, 'ROOT', tmp_path)
    for name in ('collect', 'train', 'benchmark'):
        monkeypatch.setattr(distill, name, lambda *args, **kwargs: pytest.fail('Ran despite existing output'))
    monkeypatch.setattr(sys, 'argv', ['mario-distill', command, flag, str(target)])
    with pytest.raises(SystemExit) as exit_info:
        distill.main()
    assert exit_info.value.code == 2 and 'already exists' in capsys.readouterr().err
    assert target.read_bytes() == b'keep me'


def test_distill_cli_benchmark_accepts_existing_output(tmp_path, monkeypatch):
    # Only collect/train are guarded; benchmark reads the existing --output student.
    calls = []
    output = tmp_path/'student.pt'
    output.write_bytes(b'existing student')
    monkeypatch.setattr(distill, 'ROOT', tmp_path)
    monkeypatch.setattr(distill, 'benchmark', lambda *args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', ['mario-distill', 'benchmark', '--output', str(output)])
    distill.main()
    assert calls == [(tmp_path/'models/distillation/frames.pt', output, 'auto')]


@pytest.mark.parametrize(('width', 'pool'), [(1, 8), (2, 8), (.5, 4)])
def test_student_width_and_pool_roundtrip(tmp_path, width, pool):
    torch.set_num_threads(1)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[100:140, 60:80] = (200, 50, 20)
    model = StudentCNN(pool, width).eval()
    assert any(isinstance(layer, torch.nn.MaxPool2d) for layer in model.net)
    assert model.net[0].out_channels == round(16*width)
    assert model.net[-3].in_features == round(64*width)*pool*pool
    with torch.no_grad():
        model.target_mean.normal_(generator=torch.Generator().manual_seed(3))
        expected = model(preprocess(frame))[0]
    assert expected.shape == (768,) and torch.isfinite(expected).all()
    path = tmp_path/'student.pt'
    save_student(model, path)
    payload = torch.load(path, weights_only=True)
    assert (payload['pool'], payload['arch'], payload['width']) == (pool, 'impala', float(width))
    student = StudentEncoder(path)
    assert (student.model.pool, student.model.width) == (pool, float(width))
    torch.testing.assert_close(student.encode_frame(frame), expected, rtol=0., atol=0.)


def test_student_payload_without_width_loads_as_width_1(tmp_path):
    model = StudentCNN(4).eval()
    path = tmp_path/'nowidth.pt'
    torch.save({'format': 'mario-student-v1', 'metadata': action_cache_metadata(70), 'pool': 4, 'arch': 'impala',
                'weights': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'metrics': {}}, path)
    student = StudentEncoder(path)
    assert (student.model.pool, student.model.width) == (4, 1.)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    with torch.no_grad():
        torch.testing.assert_close(student.encode_frame(frame), model(preprocess(frame))[0], rtol=0., atol=0.)


@pytest.mark.parametrize('arch', ['missing', 'small', 'resnet', None])
def test_student_encoder_rejects_non_impala_payloads(tmp_path, arch):
    path = tmp_path/'student.pt'
    save_student(StudentCNN(4), path)
    payload = torch.load(path, weights_only=True)
    if arch == 'missing':
        del payload['arch']  # Files from before the arch key were the plain CNN.
    else:
        payload['arch'] = arch
    torch.save(payload, path)
    with pytest.raises(ValueError, match='IMPALA'):
        StudentEncoder(path)


@pytest.mark.parametrize('width', [0, -1, .01, float('nan'), float('inf')])
def test_invalid_student_width_raises(width):
    with pytest.raises(ValueError):
        StudentCNN(4, width)


def test_smallest_valid_student_width_builds():
    # round(16*width) >= 1 is the floor; 0.5 is the documented valid lower case.
    for width in (.5, 1/16):
        model = StudentCNN(4, width).eval()
        with torch.no_grad():
            assert model(preprocess(np.zeros((240, 256, 3), dtype=np.uint8))).shape == (1, 768)


# --- CLI checkpoint guard and per-command defaults -------------------------------------------

def run_defaults(monkeypatch, tmp_path, argv, payload=None):
    """Run cli.main with fakes; report encoder choice, display, sleeps and started-event mode."""
    import mario_embedding_agent.student as student_module
    seen = {'frozen': 0, 'student': [], 'display': 0, 'sleeps': 0, 'calibrated': [], 'loads': []}

    class Frozen(Encoder):
        def __init__(self, *args, **kwargs):
            seen['frozen'] += 1

    class Student(Encoder):
        fingerprint = 'fake-student'
        def __init__(self, path, device='auto'):
            seen['student'].append(path)

    class FakeDisplay:
        def __init__(self):
            seen['display'] += 1
        def draw(self, frame, policy_frame, lines):
            return True
        def close(self):
            pass

    class Env:
        steps = 0
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass

    def fake_step(environment, command):
        environment.steps += 1
        return np.zeros((240, 256, 3), dtype=np.uint8), environment.steps == 2

    def calibrate(encoder, path):
        seen['calibrated'].append(path)
        return torch.zeros(768), torch.ones(768)

    def load(path):
        seen['loads'].append(path)
        return payload

    def sleep(_):
        seen['sleeps'] += 1

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Frozen)
    monkeypatch.setattr(student_module, 'StudentEncoder', Student)
    monkeypatch.setattr(student_module, 'calibrate_policy_features', calibrate)
    monkeypatch.setattr(cli, 'Display', FakeDisplay)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: {})
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0., 0., [1.] + [0.] * 17))
    monkeypatch.setattr(cli.checkpoint, 'load', load)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: None)
    monkeypatch.setattr(cli.time, 'sleep', sleep)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', *argv])
    cli.main()
    command = argv[0]
    events = [json.loads(line) for line in (tmp_path/f'runs/{command}.jsonl').read_text().splitlines()]
    seen['mode'] = next(event for event in events if event['event'] == 'started')['mode']
    return seen


def existing_checkpoint(tmp_path):
    path = tmp_path/'checkpoints/latest.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'original checkpoint bytes')
    return path


def test_fresh_train_refuses_to_overwrite_existing_checkpoint(monkeypatch, tmp_path, capsys):
    path = existing_checkpoint(tmp_path)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Encoder built before guard'))
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: pytest.fail('Checkpoint overwritten'))
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--teacher', '--total-frames', '1'])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert '--resume' in capsys.readouterr().err
    assert path.read_bytes() == b'original checkpoint bytes'


def test_resume_continues_the_existing_checkpoint(monkeypatch, tmp_path):
    path = existing_checkpoint(tmp_path)
    payload = legacy_payload({})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT)
    seen = run_defaults(monkeypatch, tmp_path, ['train', '--teacher', '--resume', '--total-frames', '1'], payload)
    assert seen['loads'] == [path]
    assert path.read_bytes() == b'original checkpoint bytes'  # save is faked; the guard didn't trip.


@pytest.mark.parametrize('command', [['play'], ['evaluate'], ['train', '--resume']])
def test_play_evaluate_and_resume_require_existing_checkpoint(monkeypatch, tmp_path, capsys, command):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Encoder built before guard'))
    monkeypatch.setattr(cli.checkpoint, 'load', lambda *args: pytest.fail('Loaded a missing checkpoint'))
    monkeypatch.setattr(sys, 'argv', ['mario-agent', *command, '--teacher', '--checkpoint', str(tmp_path/'missing.pt')])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert 'does not exist' in capsys.readouterr().err


@pytest.mark.parametrize(('argv', 'mode', 'sleeps', 'display'), [
    (['train', '--total-frames', '3'], 'sync', False, False),
    (['train', '--total-frames', '3', '--show'], 'sync', False, True),
    (['train', '--total-frames', '3', '--mode', 'realtime', '--fps', '30'], 'realtime', True, False),
    (['train', '--total-frames', '3', '--mode', 'realtime'], 'realtime', True, False),  # fps defaults to 60.
    (['evaluate', '--episodes', '1'], 'sync', False, False),
    (['evaluate', '--episodes', '1', '--show'], 'sync', False, True),
    (['play', '--total-frames', '3'], 'sync', True, True),
    (['play', '--total-frames', '3', '--headless'], 'sync', True, False),
    (['play', '--total-frames', '3', '--fps', '0'], 'sync', False, True),
])
def test_cli_defaults_per_command(monkeypatch, tmp_path, argv, mode, sleeps, display):
    existing_checkpoint(tmp_path)
    if argv[0] == 'train':
        (tmp_path/'checkpoints/latest.pt').unlink()
    seen = run_defaults(monkeypatch, tmp_path, [*argv, '--teacher'], legacy_payload({}))
    assert seen['mode'] == mode
    assert (seen['sleeps'] > 0) == sleeps  # fps 0 (train/evaluate default) never sleeps; play paces at 60.
    assert seen['display'] == int(display)


def test_student_encoder_is_default_and_teacher_flag_overrides(monkeypatch, tmp_path):
    seen = run_defaults(monkeypatch, tmp_path, ['train', '--total-frames', '1'])
    assert seen['student'] == [tmp_path/'models/distillation/student.pt'] and seen['frozen'] == 0
    assert seen['calibrated'] == [tmp_path/'models/distillation/frames.pt']  # Fresh student MLP calibrates.
    (tmp_path/'checkpoints/latest.pt').unlink(missing_ok=True)
    seen = run_defaults(monkeypatch, tmp_path, ['train', '--total-frames', '1', '--teacher'])
    assert seen['student'] == [] and seen['frozen'] == 1 and seen['calibrated'] == []


def test_prepare_always_uses_teacher(monkeypatch, tmp_path):
    import mario_embedding_agent.student as student_module
    built = []
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'download_model', lambda directory: None)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: built.append('teacher') or Encoder())
    monkeypatch.setattr(student_module, 'StudentEncoder', lambda *args, **kwargs: pytest.fail('Student used for prepare'))
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    for extra in ([], ['--student-encoder', str(tmp_path/'s.pt')]):
        monkeypatch.setattr(sys, 'argv', ['mario-agent', 'prepare', *extra])
        cli.main()
    assert built == ['teacher', 'teacher']


@pytest.mark.parametrize('case', ['fresh_custom', 'fresh_default', 'resume'])
def test_student_mlp_calibrates_from_calibration_frames_only_when_fresh(monkeypatch, tmp_path, case):
    import mario_embedding_agent.student as student_module
    from mario_embedding_agent.policy import ActorCritic
    path = tmp_path/'student.pt'
    save_student(StudentCNN(), path)
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
    argv = ['mario-agent', 'train', '--student-encoder', str(path), 
            '--total-frames', '1', '--policy-architecture', 'mlp']
    if case != 'fresh_default':
        argv += ['--calibration-frames', str(custom)]
    if case == 'resume':
        argv += ['--resume']
        (tmp_path/'checkpoints').mkdir()
        (tmp_path/'checkpoints/latest.pt').write_bytes(b'placeholder')  # load is faked
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    expected = {'fresh_custom': [custom], 'fresh_default': [tmp_path/'models/distillation/frames.pt'],
                'resume': []}[case]
    assert calls == expected
    assert saved[-1]['encoder_backend'] == 'student' and saved[-1]['policy_architecture'] == 'mlp'




@pytest.mark.parametrize(('saved_backend', 'flags', 'message'), [
    ('embeddinggemma2', [], r'trained with the teacher \(pass --teacher\)'),
    ('student', ['--teacher'], r'trained with the student encoder \(omit --teacher\)'),
])
def test_resume_backend_mismatch_names_the_teacher_flag(monkeypatch, tmp_path, saved_backend, flags, message):
    existing_checkpoint(tmp_path)
    payload = legacy_payload({})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT,
                             encoder_backend=saved_backend,
                             student_fingerprint='fake-student' if saved_backend == 'student' else None)
    with pytest.raises(ValueError, match=message):
        run_defaults(monkeypatch, tmp_path, ['train', '--resume', '--total-frames', '1', *flags], payload)


# --- Adaptive entropy coefficient -------------------------------------------------------------

def scripted_update(results):
    """Fake PPO update returning scripted metrics; records the entropy_coef each call used."""
    used = []
    def update(policy, optimizer, rollout, bootstrap, *, entropy_coef, **kwargs):
        used.append(entropy_coef)
        return dict(results[len(used)-1]) if len(used) <= len(results) else {}
    return update, used


def test_entropy_coef_adapts_toward_target_and_clamps(monkeypatch, tmp_path):
    import math
    # The adaptation reads the actor-only entropy; recovery-only updates report it as None.
    results = [{'actor_samples': 4, 'actor_entropy': .1},                     # below target 0.5 -> rises
               {'actor_samples': 0, 'actor_entropy': None, 'entropy': -10.},  # recovery-only -> unchanged
               {'actor_samples': 4, 'actor_entropy': -10.},                   # far below -> clamps at 0.1
               {'actor_samples': 4, 'actor_entropy': 5.},                     # above -> falls
               {'actor_samples': 4, 'actor_entropy': 5.},                     # below floor -> clamps at 0.01
               {'actor_samples': 4, 'entropy': -10.}]                         # no actor entropy -> unchanged
    update, used = scripted_update(results)
    _, events, _, saved = run(monkeypatch, tmp_path, ['train', '--total-frames', '200', '--rollout', '2'],
                              update=update)
    assert len(used) > len(results)
    risen = .01*math.exp(.5*(.5-.1))
    expected = [.01, risen, risen, .1, .1*math.exp(.5*(.5-5.)), .01, .01]
    assert used[:7] == pytest.approx(expected)
    logged = [event['entropy_coef'] for event in events if event['event'] == 'ppo']
    assert logged[:7] == pytest.approx(expected)  # Logs the coefficient that update used.
    config, state = saved[-1]
    assert (config['entropy_target'], config['entropy_coef_max'], config['entropy_adapt_rate']) == (.5, .1, .5)
    assert config['entropy_coef'] == .01  # Floor stays the configured base value.
    assert state['entropy_coef'] == pytest.approx(.01)


def test_entropy_coef_persists_in_checkpoint_and_resumes(monkeypatch, tmp_path):
    import math
    update, used = scripted_update([{'actor_samples': 4, 'actor_entropy': .3}])
    _, _, _, saved = run(monkeypatch, tmp_path, ['train', '--total-frames', '60', '--rollout', '2'],
                         update=update)
    assert used[0] == .01 and used[1] == pytest.approx(.01*math.exp(.1))
    assert saved[-1][1]['entropy_coef'] == pytest.approx(.01*math.exp(.1))
    for name in ('resume', 'legacy'):
        (tmp_path/name).mkdir()
    payload = legacy_payload({'entropy_coef': .05})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT)
    update, used = scripted_update([{'actor_samples': 4, 'actor_entropy': .5}])  # At target: unchanged.
    _, _, _, saved = run(monkeypatch, tmp_path/'resume', ['train', '--total-frames', '60', '--rollout', '2', '--resume'],
                         payload=payload, update=update)
    assert used[0] == pytest.approx(.05) and used[1] == pytest.approx(.05)
    assert saved[-1][1]['entropy_coef'] == pytest.approx(.05)
    # Older checkpoints without the key start from the base coefficient.
    update, used = scripted_update([])
    run(monkeypatch, tmp_path/'legacy', ['train', '--total-frames', '60', '--rollout', '2', '--resume'],
        payload=legacy_payload({}), update=update)
    assert used[0] == .01


# --- Best checkpoint ---------------------------------------------------------------------------

def wilson_lower_bound(clears, n):
    import math
    rate = clears/n
    return (rate + 1/(2*n) - math.sqrt(rate*(1-rate)/n + 1/(4*n*n))) / (1 + 1/n)


def test_best_checkpoint_requires_full_window_and_a_clear(monkeypatch, tmp_path):
    # No new ground (so no restart snapshots); a death every 10 steps: 45 full-start failures.
    env, events, _, saved = run(monkeypatch, tmp_path, ['train', '--total-frames', '455', '--rollout', '2'],
                                death_at=set(range(10, 451, 10)))
    assert not any(event['event'] == 'best_checkpoint' for event in events)
    assert all(path == tmp_path/'checkpoints/latest.pt' for path in env.save_paths)
    state = saved[-1][1]
    assert state['recent_full_start_clears'] == [0]*40 and state['best_clear_rate'] == -1.


def test_best_checkpoint_saved_on_improvement_not_after_decline(monkeypatch, tmp_path):
    # 38 counted deaths, clear (counted 1), continuation death (not counted), clear (counted 1):
    # window full with 2 clears. Then 39 more counted deaths push the first clear out (1 clear).
    deaths = set(range(10, 381, 10)) | {390, 400} | set(range(410, 791, 10))
    clears = {385, 395}
    env, events, _, saved = run(monkeypatch, tmp_path, ['train', '--total-frames', '795', '--rollout', '2'],
                                death_at=deaths, clear_at=clears)
    best = [event for event in events if event['event'] == 'best_checkpoint']
    assert best, 'expected a best checkpoint once the window had clears'
    two = wilson_lower_bound(2, 40)
    assert all(event['clear_rate'] == pytest.approx(2/40) for event in best)
    assert all(event['lower_bound'] == pytest.approx(two) for event in best)
    assert all(event['path'] == str(tmp_path/'checkpoints/best.pt') for event in best)
    assert best[0]['clears'] == 2 and best[0]['frames'] >= 395
    # Every best save immediately follows a latest save that already carries the new score.
    for index, path in enumerate(env.save_paths):
        if path == tmp_path/'checkpoints/best.pt':
            assert env.save_paths[index-1] == tmp_path/'checkpoints/latest.pt'
            assert saved[index-1][1]['best_clear_rate'] == pytest.approx(two)
            assert saved[index][1]['best_clear_rate'] == pytest.approx(two)
    assert len(best) > 1  # Ties (same window score) favour the newer policy.
    assert env.save_paths.count(tmp_path/'checkpoints/best.pt') == len(best)
    state = saved[-1][1]
    assert len(state['recent_full_start_clears']) == 40 and sum(state['recent_full_start_clears']) == 1
    assert wilson_lower_bound(1, 40) < two
    assert state['best_clear_rate'] == pytest.approx(two)  # Decline never overwrote best.
    # No best saves once the first clear left the window (score dropped).
    drop = 410 + 38*10
    assert best[-1]['frames'] < drop


def test_best_path_follows_custom_checkpoint_stem(monkeypatch, tmp_path):
    custom = tmp_path/'runs-a/policy.pt'
    env, _, _, _ = run(monkeypatch, tmp_path, ['train', '--total-frames', '398', '--rollout', '100000',
                                               '--checkpoint', str(custom)],
                       death_at=set(range(10, 391, 10)), clear_at={395})
    assert env.save_paths == [custom, tmp_path/'runs-a/policy-best.pt']


# --- Checkpoint path resolution ----------------------------------------------------------------

@pytest.mark.parametrize('command', ['play', 'evaluate'])
@pytest.mark.parametrize('best_exists', [True, False])
def test_play_and_evaluate_prefer_best_checkpoint(monkeypatch, tmp_path, command, best_exists):
    latest = existing_checkpoint(tmp_path)
    best = tmp_path/'checkpoints/best.pt'
    if best_exists:
        best.write_bytes(b'best')
    argv = [command, '--teacher'] + (['--episodes', '1'] if command == 'evaluate' else ['--total-frames', '3'])
    seen = run_defaults(monkeypatch, tmp_path, argv, legacy_payload({}))
    assert seen['loads'] == [best if best_exists else latest]


@pytest.mark.parametrize('command', ['play', 'evaluate', 'train'])
def test_explicit_checkpoint_wins_over_best(monkeypatch, tmp_path, command):
    existing_checkpoint(tmp_path)
    (tmp_path/'checkpoints/best.pt').write_bytes(b'best')
    explicit = tmp_path/'mine.pt'
    explicit.write_bytes(b'mine')
    extra = {'evaluate': ['--episodes', '1'], 'play': ['--total-frames', '3'],
             'train': ['--total-frames', '3', '--resume']}[command]
    seen = run_defaults(monkeypatch, tmp_path, [command, '--teacher', '--checkpoint', str(explicit), *extra],
                        legacy_payload({}))
    assert seen['loads'] == [explicit]


def test_train_resume_ignores_best_checkpoint(monkeypatch, tmp_path):
    latest = existing_checkpoint(tmp_path)
    (tmp_path/'checkpoints/best.pt').write_bytes(b'best')
    seen = run_defaults(monkeypatch, tmp_path, ['train', '--teacher', '--total-frames', '3', '--resume'],
                        legacy_payload({}))
    assert seen['loads'] == [latest]


def test_fresh_train_refused_when_only_best_checkpoint_exists(monkeypatch, tmp_path, capsys):
    (tmp_path/'checkpoints').mkdir()
    best = tmp_path/'checkpoints/best.pt'
    best.write_bytes(b'best policy')
    custom_best = tmp_path/'other/run-best.pt'
    custom_best.parent.mkdir()
    custom_best.write_bytes(b'other best')
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Encoder built before guard'))
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: pytest.fail('Checkpoint written'))
    for argv, existing in [([], best), (['--checkpoint', str(tmp_path/'other/run.pt')], custom_best)]:
        monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--teacher', '--total-frames', '1', *argv])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
        assert str(existing) in capsys.readouterr().err
    assert best.read_bytes() == b'best policy' and custom_best.read_bytes() == b'other best'


def test_fresh_train_refusal_lists_both_existing_files(monkeypatch, tmp_path, capsys):
    latest = existing_checkpoint(tmp_path)
    best = tmp_path/'checkpoints/best.pt'
    best.write_bytes(b'best')
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'train', '--teacher', '--total-frames', '1'])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    err = capsys.readouterr().err
    assert exit_info.value.code == 2 and str(latest) in err and str(best) in err


def test_started_event_logs_restored_entropy_coef(monkeypatch, tmp_path):
    payload = legacy_payload({'entropy_coef': .037})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT)
    _, events, _, _ = run(monkeypatch, tmp_path, train_argv(1) + ['--resume'], payload=payload)
    started = next(event for event in events if event['event'] == 'started')
    assert started['entropy_coef'] == pytest.approx(.037)
    (tmp_path/'fresh').mkdir()
    _, events, _, _ = run(monkeypatch, tmp_path/'fresh', train_argv(1))
    started = next(event for event in events if event['event'] == 'started')
    assert started['entropy_coef'] == .01


# --- ppo.update actor_entropy --------------------------------------------------------------------

def test_ppo_actor_entropy_is_final_policy_entropy_on_actor_samples_only():
    from torch.distributions import Categorical
    from mario_embedding_agent.environment import ACTIONS
    from mario_embedding_agent.policy import ActorCritic
    from mario_embedding_agent.ppo import Transition, update
    torch.manual_seed(0)
    policy = ActorCritic(torch.randn(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bias = torch.zeros(len(ACTIONS)); bias[5] = 8.
    transitions = []
    for i in range(8):
        feature = torch.randn(768)
        action_bias = bias if i % 2 else None
        action, log_prob, value, _ = policy.act(feature, action_bias=action_bias)
        transitions.append(Transition(feature, action, log_prob, value, float(i), i == 7, action_bias))
    metrics = update(policy, optimizer, transitions, 0., epochs=2, minibatch_size=4, target_kl=None)
    assert metrics['actor_samples'] == 4 and metrics['recovery_samples'] == 4
    actor = [item for item in transitions if item.action_bias is None]
    with torch.no_grad():
        logits, _ = policy(torch.stack([item.features for item in actor]),
                           action_bias=torch.zeros(len(actor), len(ACTIONS)),
                           previous_features=torch.zeros(len(actor), 768))
        expected = float(Categorical(logits=logits).entropy().mean())
    assert metrics['actor_entropy'] == pytest.approx(expected, rel=1e-5)


def test_ppo_actor_entropy_none_for_recovery_only_rollout():
    from mario_embedding_agent.environment import ACTIONS
    from mario_embedding_agent.policy import ActorCritic
    from mario_embedding_agent.ppo import Transition, update
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bias = torch.zeros(len(ACTIONS)); bias[5] = 8.
    transitions = []
    for i in range(8):
        feature = torch.randn(768)
        action, log_prob, value, _ = policy.act(feature, action_bias=bias)
        transitions.append(Transition(feature, action, log_prob, value, 1., i == 7, bias))
    metrics = update(policy, optimizer, transitions, 0., epochs=1, minibatch_size=8)
    assert metrics['actor_samples'] == 0 and metrics['actor_entropy'] is None


def test_ppo_actor_entropy_reported_after_kl_early_stop():
    from torch.distributions import Categorical
    from mario_embedding_agent.environment import ACTIONS
    from mario_embedding_agent.policy import ActorCritic
    from mario_embedding_agent.ppo import Transition, update
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    feature = torch.ones(768)
    action, log_prob, value, _ = policy.act(feature, sample=False)
    transitions = [Transition(feature, action, log_prob, value, 1., True) for _ in range(8)]
    with torch.no_grad():
        policy.actor.bias[action] += 8.
        logits, _ = policy(feature)
    metrics = update(policy, optimizer, transitions, 0., epochs=4, minibatch_size=8, target_kl=.02)
    assert metrics['stopped_early'] and metrics['optimizer_steps'] == 0
    # The unchanged policy's true (low) entropy is reported, not a missing/zero minibatch value.
    assert metrics['actor_entropy'] == pytest.approx(float(Categorical(logits=logits).entropy()), rel=1e-3)


def test_kl_early_stop_without_actor_entropy_leaves_coefficient(monkeypatch, tmp_path):
    # An update that early-stopped and reported only minibatch 'entropy' (or None actor entropy)
    # must not move the coefficient.
    update, used = scripted_update([{'actor_samples': 4, 'stopped_early': True, 'entropy': 0.},
                                    {'actor_samples': 4, 'stopped_early': True, 'actor_entropy': None}])
    run(monkeypatch, tmp_path, ['train', '--total-frames', '40', '--rollout', '2'], update=update)
    assert used[:3] == [.01, .01, .01]


# --- play starts from the level start --------------------------------------------------------

def in_progress_payload():
    from mario_embedding_agent.environment import COMMANDS
    payload = legacy_payload({'episode_commands': [COMMANDS.index(('right',))]*3,
                              'visual_position_px': 40., 'visual_frontier_px': 50.,
                              'stall_frames': 300, 'stall_frontier_px': 50., 'stagnant_decisions': 5,
                              'recovery_active': True, 'recovery_action_index': 2,
                              'recovery_progress_window': [1., 2.], 'episode_from_start': False,
                              'previous_features': torch.ones(768), 'clears': 4, 'stalls': 6})
    payload['config'].update(reward_version='visual-frontier-v5', stall_timeout_frames=TIMEOUT)
    return payload


def test_play_discards_in_progress_episode_but_keeps_counters(monkeypatch, tmp_path):
    env, events, _, saved = run(monkeypatch, tmp_path, ['play', '--headless', '--fps', '0', '--total-frames', '5'],
                                payload=in_progress_payload())
    assert env.steps == 5  # No replay steps before the first decision.
    assert env.resets == [3]
    started = next(event for event in events if event['event'] == 'started')
    assert started['visual_frontier_px'] == 0. and started['visual_position_px'] == 0.
    assert started['recovery_active'] is False and started['recovery_action_index'] == 0
    assert started['stagnant_decisions'] == 0
    assert started['stall_frames'] == 0
    assert (started['frames'], started['updates'], started['deaths'], started['clears'], started['stalls']) == \
        (100, 5, 2, 4, 6)
    first = next(event for event in events if event['event'] == 'decision')
    # ('started' reports the configured recovery_progress_window size, so check the state here.)
    assert first['frames'] == 100 and first['recovery_action'] is None
    assert first['recovery_progress_window'] == []
    assert saved == []


def test_train_resume_still_replays_in_progress_episode(monkeypatch, tmp_path):
    env, events, _, saved = run(monkeypatch, tmp_path, train_argv(5) + ['--resume'], payload=in_progress_payload())
    assert env.steps == 3 + 5  # Three replayed commands, then five new frames.
    started = next(event for event in events if event['event'] == 'started')
    assert started['visual_frontier_px'] == 50. and started['recovery_active'] is True
    assert started['stagnant_decisions'] == 5 and started['stall_frames'] == 300
    first = next(event for event in events if event['event'] == 'decision')
    assert first['recovery_action'] is not None and first['recovery_progress_window'] == [1., 2.]
    assert len(saved[-1][1]['episode_commands']) == 3 + 5
