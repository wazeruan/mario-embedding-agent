"""Distillation integrity tests; synthetic smoke data is not gameplay evidence."""
import hashlib
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mario_embedding_agent.encoder import action_cache_metadata, prepare_actions
from mario_embedding_agent.student import SmallEncoder, StudentEncoder, preprocess, save_student
from mario_embedding_agent import distill


def dataset(path, count=64):
    frames = torch.zeros(count, 240, 256, 3, dtype=torch.uint8)
    frames[:, 0, 0, 0] = torch.arange(count, dtype=torch.uint8)
    targets = F.normalize(torch.randn(count, 768, generator=torch.Generator().manual_seed(33)), dim=1)
    groups = torch.tensor([(i % 4, i // 16) for i in range(count)])
    torch.save(dict(frames=frames, targets=targets, groups=groups,
                    metadata=action_cache_metadata(70)), path)
    return targets, groups


def test_student_normalized_embedding_roundtrip_fingerprint_and_teacher_free_inference(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    model = SmallEncoder().eval()
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    with torch.no_grad():
        expected = model(preprocess(frame))[0]
    path = tmp_path/'student.pt'
    save_student(model, path)
    monkeypatch.setattr(distill, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Teacher constructed'))
    student = StudentEncoder(path)
    actual = student.encode_frame(frame)
    assert actual.shape == (768,) and torch.isfinite(actual).all()
    assert actual.norm() == pytest.approx(1., abs=1e-6)
    assert not torch.is_inference(actual)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert student.fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not student.model.training and all(not parameter.requires_grad for parameter in student.model.parameters())
    with torch.no_grad():
        model.net[-1].bias[0] += .1
    save_student(model, path)
    assert StudentEncoder(path).fingerprint != student.fingerprint
    cache = tmp_path/'actions.pt'
    torch.save(dict(metadata=action_cache_metadata(70), embeddings=torch.ones(18, 768)), cache)
    assert prepare_actions(student, cache).shape == (18, 768)
    with pytest.raises(ValueError, match='image-only'):
        prepare_actions(student, tmp_path/'missing.pt')


@pytest.mark.parametrize('field', ['revision', 'question', 'image_tokens'])
def test_dataset_and_student_reject_teacher_metadata_mismatch(tmp_path, field):
    data_path = tmp_path/'data.pt'
    dataset(data_path)
    student_path = tmp_path/'student.pt'
    save_student(SmallEncoder(), student_path)
    for path, loader in [(data_path, distill.load_data), (student_path, StudentEncoder)]:
        payload = torch.load(path, weights_only=True)
        payload['metadata'][field] = 'wrong'
        torch.save(payload, path)
        with pytest.raises(ValueError, match='metadata'):
            loader(path)


def test_stream_group_holdout_and_train_only_target_statistics(tmp_path, monkeypatch):
    data_path, student_path = tmp_path/'data.pt', tmp_path/'student.pt'
    targets, groups = dataset(data_path)
    data, train_mask, valid_mask = distill.load_data(data_path)
    assert torch.equal(valid_mask, groups[:, 0] == 3)
    train_groups = {tuple(group) for group in groups[train_mask].tolist()}
    valid_groups = {tuple(group) for group in groups[valid_mask].tolist()}
    assert train_groups.isdisjoint(valid_groups)
    training_targets = []
    original_loss = distill.F.mse_loss
    def record_loss(predicted_residual, expected_residual):
        training_targets.append(expected_residual.detach().clone())
        return original_loss(predicted_residual, expected_residual)
    monkeypatch.setattr(distill.F, 'mse_loss', record_loss)
    report = distill.train(data_path, student_path, epochs=1)
    student = StudentEncoder(student_path)
    torch.testing.assert_close(student.model.target_mean, targets[train_mask].mean(0))
    torch.testing.assert_close(student.model.target_std, targets[train_mask].std(0).clamp_min(1e-3))
    assert not torch.allclose(student.model.target_mean, targets.mean(0))
    expected_residuals = (F.normalize(targets[train_mask], dim=1)-student.model.target_mean)/student.model.target_std
    observed = torch.cat(training_targets)
    assert observed.shape == expected_residuals.shape
    # Every target used by the optimization is a standardized training row.
    errors = (observed[:, None]-expected_residuals[None, :]).abs().amax(dim=2)
    assert errors.min(dim=1).values.max() < 1e-5
    assert {'student', 'constant_mean'}.issubset(report)
    assert report['student']['standardized_rmse'] >= 0.
    baseline = F.normalize(targets[train_mask].mean(0), dim=0).expand_as(targets[valid_mask])
    expected = ((baseline-targets[valid_mask])/student.model.target_std).square().mean().sqrt()
    assert report['constant_mean']['standardized_rmse'] == pytest.approx(float(expected), rel=1e-5)


def test_collection_exact_dedup_grouping_and_death_only_resets(tmp_path, monkeypatch):
    environments, teacher_frames = [], []
    class Env:
        def __init__(self):
            self.stream = len(environments)
            self.ticks = 0
            self.resets = []
            self.closed = False
            self.deaths = 0
            environments.append(self)
        def reset(self, seed):
            self.resets.append(seed)
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            self.closed = True
    class Teacher:
        def __init__(self, *args, **kwargs):
            assert kwargs['cache_size'] == 0
        def encode_frame(self, frame):
            teacher_frames.append(frame.tobytes())
            return F.normalize(torch.arange(1, 769).float(), dim=0)
    def step(env, command):
        assert isinstance(command, tuple), 'Environment helper consumes controller buttons, not index'
        env.ticks += 1
        frame = np.zeros((240, 256, 3), dtype=np.uint8)
        # Repeated black observations first, then stream-unique pixels.
        if env.ticks >= 30:
            frame[0, 0] = (env.stream + 1, env.ticks % 256, env.ticks // 256)
        died = env.ticks % 37 == 0
        env.deaths += int(died)
        return frame, died
    monkeypatch.setattr(distill, 'make_env', Env)
    monkeypatch.setattr(distill, 'step', step)
    monkeypatch.setattr(distill, 'FrozenEncoder', Teacher)
    path = tmp_path/'collected.pt'
    distill.collect(path, samples=128)
    data, train_mask, valid_mask = distill.load_data(path)
    hashes = [hashlib.sha256(frame.numpy().tobytes()).hexdigest() for frame in data['frames']]
    assert len(hashes) == len(set(hashes)) == len(teacher_frames) == 128
    assert valid_mask.sum() >= 16 and train_mask.sum() >= 16
    assert set(data['groups'][:, 0].tolist()) == {0, 1, 2, 3}
    assert data['groups'][:, 1].max() > 0
    assert torch.allclose(data['targets'].norm(dim=1), torch.ones(128))
    for env in environments:
        assert env.closed and len(env.resets) == 1 + env.deaths


@pytest.mark.parametrize('resume_kind', ['fresh', 'teacher', 'different_student', 'same_student'])
def test_cli_student_teacher_free_and_resume_backend_identity(monkeypatch, tmp_path, resume_kind):
    from mario_embedding_agent import cli
    from mario_embedding_agent.policy import ActorCritic
    path = tmp_path/'student.pt'
    save_student(SmallEncoder(), path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = tmp_path/'models/action-embeddings.pt'
    cache.parent.mkdir()
    torch.save(dict(metadata=action_cache_metadata(70), embeddings=torch.ones(18, 768)), cache)
    policy = ActorCritic(torch.ones(18, 768))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    old_config = {'reward_scale': 50., 'image_tokens': 70, 'seed': 0,
                  'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
                  'encoder_backend': 'embeddinggemma2' if resume_kind == 'teacher' else 'student',
                  'student_fingerprint': None if resume_kind == 'teacher' else
                      ('different' if resume_kind == 'different_student' else fingerprint)}
    payload = {'config': old_config, 'policy': policy.state_dict(),
               'optimizer': optimizer.state_dict(),
               'training_state': {'episode_seed': 0, 'episode_commands': [],
                                  'torch_rng_state': torch.get_rng_state()}}
    class Env:
        def reset(self, seed):
            return np.zeros((240, 256, 3), dtype=np.uint8), {}
        def close(self):
            pass
    saved = []
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', lambda *args, **kwargs: pytest.fail('Teacher initialized for student run'))
    monkeypatch.setattr(cli, 'make_env', Env)
    monkeypatch.setattr(cli, 'step', lambda *args: (np.zeros((240, 256, 3), dtype=np.uint8), False))
    monkeypatch.setattr(cli.checkpoint, 'load', lambda path: payload)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda path, policy, optimizer, config, state: saved.append(dict(config)))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    argv = ['mario-agent', 'train', '--student-encoder', str(path), '--headless', '--mode', 'sync', '--total-frames', '1']
    if resume_kind != 'fresh':
        argv += ['--resume', 'resume.pt']
    monkeypatch.setattr(sys, 'argv', argv)
    if resume_kind in ('teacher', 'different_student'):
        with pytest.raises(ValueError, match='configuration is incompatible'):
            cli.main()
        assert not saved
    else:
        cli.main()
        assert saved[-1]['encoder_backend'] == 'student'
        assert saved[-1]['student_fingerprint'] == fingerprint


def test_benchmark_is_embedding_fidelity_and_latency_without_gameplay_claim(tmp_path, monkeypatch):
    import json
    data_path, path = tmp_path/'data.pt', tmp_path/'student.pt'
    dataset(data_path)
    save_student(SmallEncoder().eval(), path)
    teacher = StudentEncoder(path)
    monkeypatch.setattr(distill, 'ROOT', tmp_path)
    monkeypatch.setattr(distill, 'FrozenEncoder', lambda *args, **kwargs: teacher)
    cache = tmp_path/'models/action-embeddings.pt'
    cache.parent.mkdir()
    torch.save(dict(metadata=action_cache_metadata(70), embeddings=torch.randn(18, 768)), cache)
    report = distill.benchmark(data_path, path, count=8)
    assert report['student']['cosine'] == pytest.approx(1., abs=1e-6)
    assert report['student']['standardized_rmse'] == 0.
    assert report['action_prototype_top1_agreement'] == 1.
    assert report['gameplay_evaluated'] is False
    assert report['samples'] == 8
    assert report['checkpoint_bytes'] == path.stat().st_size
    assert set(report['latency']) == {'student', 'teacher'}
    saved = json.loads(path.with_suffix('.benchmark.json').read_text())
    assert saved == report


def test_train_report_matches_saved_best_checkpoint_not_last_epoch(tmp_path, monkeypatch):
    data_path, path = tmp_path/'data.pt', tmp_path/'student.pt'
    dataset(data_path)
    scores = iter([3., 1., 2.])
    monkeypatch.setattr(distill, 'metrics', lambda *args: {
        'student': {'standardized_rmse': next(scores)}, 'constant_mean': {'standardized_rmse': 4.}})
    report = distill.train(data_path, path, epochs=3)
    payload = torch.load(path, weights_only=True)
    assert report == payload['metrics']
    assert report['student']['standardized_rmse'] == 1.
