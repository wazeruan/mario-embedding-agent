"""Small image-only student in the pinned teacher's embedding space."""
import hashlib
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


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 24, 5, 2, 2), nn.GELU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 768))
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
        self.model = SmallEncoder().to(self.device)
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
