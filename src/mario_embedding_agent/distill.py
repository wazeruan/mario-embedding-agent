"""Collect real RGB frames, distill a small encoder, and benchmark held-out streams."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .actions import ACTIONS
from .encoder import FrozenEncoder, action_cache_metadata
from .environment import make_env, step
from .student import SmallEncoder, StudentEncoder, preprocess, save_student

ROOT = Path(__file__).resolve().parents[2]


def collect(path, samples=1024, seed=0, device='auto'):
    teacher = FrozenEncoder(ROOT/'models/embeddinggemma-2', device=device, cache_size=0)
    envs = [make_env() for _ in range(4)]
    rng = np.random.default_rng(seed)
    frames, targets, groups = [], [], []
    seen = set()
    episodes = [0] * 4
    # Names, rather than numeric game state, define collection controller choices.
    names = ['right', 'run_right', 'right_jump_short', 'right_jump_long',
             'run_right_jump_short', 'run_right_jump_long', 'left', 'down']
    pools = [names[:6], names[:4], names[2:], names]
    observations = []
    try:
        for index, env in enumerate(envs):
            observation, _ = env.reset(seed=seed + index)
            observations.append(observation)
        for turn in range(samples * 100):
            stream = turn % 4
            name = str(rng.choice(pools[stream]))
            action = next(a for a in ACTIONS if a.name == name)
            dead = False
            for command in action.commands():
                frame, dead = step(envs[stream], command)
                observations[stream] = frame
                if dead:
                    break
            frame = observations[stream]
            key = hashlib.sha256(frame.tobytes()).digest()
            if key not in seen:
                seen.add(key)
                frames.append(frame.copy())
                targets.append(teacher.encode_frame(frame))
                groups.append((stream, episodes[stream]))
                if len(frames) % 64 == 0:
                    print(f'Collected {len(frames)}/{samples} unique real frames', flush=True)
            if dead:
                episodes[stream] += 1
                observations[stream], _ = envs[stream].reset(seed=seed + stream + episodes[stream]*4)
            if len(frames) >= samples:
                break
        if len(frames) < samples:
            raise ValueError('Insufficient unique frames within collection budget.')
    finally:
        for env in envs:
            env.close()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'frames': torch.from_numpy(np.stack(frames)), 'targets': torch.stack(targets),
                'groups': torch.tensor(groups), 'metadata': action_cache_metadata(70)}, path)


def load_data(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('metadata') != action_cache_metadata(70):
        raise ValueError('Dataset teacher metadata is incompatible.')
    train = data['groups'][:, 0] != 3
    valid = ~train
    if train.sum() < 16 or valid.sum() < 16:
        raise ValueError('Need at least 16 training and held-out-stream frames.')
    return data, train, valid


def metrics(predicted, target, mean, std):
    baseline = F.normalize(mean, dim=0).expand_as(target)
    def measure(value):
        cosine = F.cosine_similarity(value, target)
        return {'cosine': float(cosine.mean()), 'cosine_p05': float(torch.quantile(cosine, .05)),
                'standardized_rmse': float(((value-target)/std).square().mean().sqrt())}
    teacher_distances = torch.pdist(target)
    student_distances = torch.pdist(predicted)
    correlation = None
    if len(teacher_distances) > 1 and teacher_distances.std() > 0 and student_distances.std() > 0:
        correlation = float(torch.corrcoef(torch.stack((teacher_distances, student_distances)))[0, 1])
    return {'student': measure(predicted), 'constant_mean': measure(baseline),
            'pairwise_distance_correlation': correlation}


def train(data_path, output, epochs=60, seed=0, pool=8):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    data, train_mask, valid_mask = load_data(data_path)
    inputs = preprocess(data['frames'])
    targets = F.normalize(data['targets'].float(), dim=1)
    model = SmallEncoder(pool)
    model.target_mean.copy_(targets[train_mask].mean(0))
    model.target_std.copy_(targets[train_mask].std(0).clamp_min(1e-3))
    residuals = (targets-model.target_mean)/model.target_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_indices = train_mask.nonzero().flatten()
    best = float('inf')
    best_report = None
    for epoch in range(epochs):
        model.train()
        for indices in train_indices[torch.randperm(len(train_indices))].split(64):
            loss = F.mse_loss(model.residual(inputs[indices]), residuals[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            report = metrics(model(inputs[valid_mask]), targets[valid_mask], model.target_mean, model.target_std)
        score = report['student']['standardized_rmse']
        if score < best:
            best = score
            best_report = report
            save_student(model, output, report)
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f'Epoch {epoch+1}/{epochs}: {json.dumps(report)}', flush=True)
    print(f'Saved best held-out checkpoint: {json.dumps(best_report)}', flush=True)
    return best_report


def benchmark(data_path, output, device='auto', count=32):
    torch.set_num_threads(1)
    data, _, valid = load_data(data_path)
    heldout = data['frames'][valid]
    indices = torch.linspace(0, len(heldout)-1, min(count, len(heldout))).long()
    frames = heldout[indices].numpy()
    student = StudentEncoder(output)
    teacher = FrozenEncoder(ROOT/'models/embeddinggemma-2', device=device, cache_size=0)
    latency = {}
    predictions = []
    for name, encoder in [('student', student), ('teacher', teacher)]:
        encoder.encode_frame(frames[0])
        times = []
        vectors = []
        for frame in frames:
            start = time.perf_counter()
            vectors.append(encoder.encode_frame(frame))
            times.append((time.perf_counter()-start)*1000)
        latency[name] = {'device': encoder.device, 'median_ms': float(np.median(times)),
                         'p95_ms': float(np.quantile(times, .95))}
        predictions.append(torch.stack(vectors))
    report = metrics(predictions[0], predictions[1], student.model.target_mean, student.model.target_std)
    action_path = ROOT/'models/action-embeddings.pt'
    if action_path.exists():
        from .encoder import prepare_actions
        actions = prepare_actions(student, action_path)
        report['action_prototype_top1_agreement'] = float(((predictions[0] @ actions.T).argmax(1) ==
                                                         (predictions[1] @ actions.T).argmax(1)).float().mean())
    report.update(latency=latency, parameters=sum(p.numel() for p in student.model.parameters()),
                  samples=len(frames), checkpoint_bytes=Path(output).stat().st_size,
                  gameplay_evaluated=False)
    Path(output).with_suffix('.benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'train', 'benchmark'])
    parser.add_argument('--data', type=Path, default=ROOT/'models/distillation/frames.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'models/distillation/student.pt')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=8, help='Student spatial pooling grid (train only)')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.samples < 64 or args.epochs < 1:
        parser.error('samples >=64 and epochs >=1 required')
    if args.command == 'collect':
        collect(args.data, args.samples, args.seed, args.device)
    elif args.command == 'train':
        train(args.data, args.output, args.epochs, args.seed, args.pool)
    else:
        benchmark(args.data, args.output, args.device)
