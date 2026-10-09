"""Small image-only student in the pinned teacher's embedding space."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .encoder import action_cache_metadata


def preprocess(frames):
    frames = torch.as_tensor(np.asarray(frames))
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.dtype != torch.uint8 or frames.shape[1:] != (240, 256, 3):
        raise ValueError('Expected original 240x256 RGB uint8 screenshots.')
    return F.interpolate(frames.permute(0, 3, 1, 2).float() / 255,
                         size=(120, 128), mode='bilinear', align_corners=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, 1, 1),
                                  nn.ReLU(), nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, inputs):
        return inputs + self.conv(inputs)


class StudentCNN(nn.Module):
    """IMPALA CNN (Espeholt et al., 2018) predicting the teacher's 768-d embedding.

    Each stage is conv, max-pool and two residual blocks. It replaced a plain
    three-layer CNN after giving more level clears in PPO on two seeds.
    """
    def __init__(self, pool=8, width=1):
        super().__init__()
        if not math.isfinite(width) or round(16*width) < 1:
            raise ValueError('Student width must be finite and large enough for one channel per layer.')
        self.pool, self.width = int(pool), float(width)
        channels = [round(c*self.width) for c in (16, 32, 64)]
        layers, previous = [], 3
        for out in channels:
            layers += [nn.Conv2d(previous, out, 3, 1, 1), nn.MaxPool2d(3, 2, 1),
                       ResidualBlock(out), ResidualBlock(out)]
            previous = out
        self.net = nn.Sequential(*layers, nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.pool, self.pool)), nn.Flatten(),
            nn.Linear(channels[-1]*self.pool*self.pool, 256), nn.GELU(), nn.Linear(256, 768))
        self.register_buffer('target_mean', torch.zeros(768))
        self.register_buffer('target_std', torch.ones(768))

    def residual(self, inputs):
        return self.net(inputs)

    def forward(self, inputs):
        return F.normalize(self.target_mean + self.target_std * self.residual(inputs), dim=-1)


def save_student(model, path, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'format': 'mario-student-v1', 'metadata': action_cache_metadata(70),
                'pool': model.pool, 'arch': 'impala', 'width': model.width,
                'weights': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'metrics': metrics or {}}, temporary)
    temporary.replace(path)


class StudentEncoder:
    def __init__(self, path, device='auto'):
        path = Path(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != 'mario-student-v1' or payload.get('metadata') != action_cache_metadata(70):
            raise ValueError('Student teacher metadata is incompatible.')
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        self.device = 'cpu' if device == 'auto' else device
        self.image_tokens = 70
        if payload.get('arch') != 'impala':
            raise ValueError('Only IMPALA student encoders are supported; distill one with mario-distill train.')
        self.model = StudentCNN(payload['pool'], payload.get('width', 1)).to(self.device)
        self.model.load_state_dict(payload['weights'])
        self.model.eval().requires_grad_(False)

    def encode_frame(self, frame):
        with torch.inference_mode():
            result = self.model(preprocess(frame).to(self.device)).cpu()[0]
        if not torch.isfinite(result).all() or result.norm() == 0:
            raise ValueError('Student returned invalid features.')
        return result.clone()

    def encode_actions(self):
        raise ValueError('Student is image-only. Run mario-agent prepare with the teacher first.')


def calibrate_policy_features(encoder, data_path):
    """Measure normalized student outputs on training streams only, once before PPO."""
    from .distill import load_data
    data, train_mask, _ = load_data(data_path)
    features = torch.stack([encoder.encode_frame(frame.numpy()) for frame in data['frames'][train_mask]])
    return features.mean(0), features.std(0).clamp_min(1e-3)
