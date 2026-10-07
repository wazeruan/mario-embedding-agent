"""Frozen, locally downloaded image/text encoder and immutable action embeddings."""
from collections import OrderedDict
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .actions import ACTIONS, action_manifest

MODEL_ID = 'google/embeddinggemma-2'
MODEL_REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
QUESTION = ('Looking at this Super Mario Bros. frame, which immediate controller action '
            'should Mario take to progress safely? Consider enemies, gaps, obstacles, '
            'Mario’s position, and whether a short or sustained jump is needed.')


def download_model(directory=Path('models/embeddinggemma-2')):
    from huggingface_hub import snapshot_download
    directory = Path(directory)
    # Avoid a stale global HF_HOME pointing at the former external disk.
    cache = directory.parent / '.hf-cache'
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             local_dir=directory, cache_dir=cache,
                             ignore_patterns=['*.md', '*.png', '*.jpg', '*.onnx', '*.gguf'])
    (directory / 'mario-model.json').write_text(json.dumps({'model_id': MODEL_ID, 'revision': MODEL_REVISION}))
    return Path(path)


class FrozenEncoder:
    def __init__(self, directory=Path('models/embeddinggemma-2'), device='auto', image_tokens=70, cache_size=32):
        from sentence_transformers import SentenceTransformer
        self.directory = Path(directory)
        marker = self.directory / 'mario-model.json'
        if not marker.exists() or json.loads(marker.read_text()) != {'model_id': MODEL_ID, 'revision': MODEL_REVISION}:
            raise ValueError('Run `uv run mario-agent prepare` to download the pinned encoder first.')
        self.device = ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
        self.image_tokens = image_tokens
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_hits = self.cache_misses = 0
        self.model = SentenceTransformer(str(self.directory.resolve()), device=self.device,
                                         local_files_only=True, config_kwargs={'audio_config': None},
                                         model_kwargs={'torch_dtype': self.dtype})
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _validate(embeddings, count):
        embeddings = embeddings.detach().to('cpu', torch.float32)
        if embeddings.shape != (count, 768) or not torch.isfinite(embeddings).all() or (embeddings.norm(dim=1) == 0).any():
            raise ValueError('Encoder returned invalid 768-dimensional features.')
        return F.normalize(embeddings, dim=1)

    @torch.inference_mode()
    def encode_actions(self):
        documents = [f'title: Mario controller action {a.name} | text: {a.description}' for a in ACTIONS]
        result = self.model.encode(documents, prompt='', convert_to_tensor=True,
                                   normalize_embeddings=True, show_progress_bar=False)
        return self._validate(result, len(ACTIONS))

    def encode_frame(self, frame):
        if frame.shape != (240, 256, 3) or frame.dtype != np.uint8:
            raise ValueError('Expected original 240x256 RGB uint8 screenshot.')
        key = frame.tobytes()
        if self.cache_size and key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].clone()
        self.cache_misses += 1
        with torch.inference_mode():
            inputs = [{'text': f'task: question answering | query: {QUESTION} <|image|>',
                       'image': [Image.fromarray(frame)]}]
            result = self.model.encode(inputs, prompt='', convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=False,
                                       processing_kwargs={'image': {'max_soft_tokens': self.image_tokens}})
            encoded = self._validate(result, 1)[0]
        # Normal detached tensor, usable as input to trainable heads.
        features = encoded.clone()
        if self.cache_size:
            self.cache[key] = features.clone()
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return features


def action_cache_metadata(image_tokens):
    return {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'actions': action_manifest(),
            'image_tokens': image_tokens, 'question': QUESTION}


def prepare_actions(encoder, path=Path('models/action-embeddings.pt')):
    path = Path(path)
    metadata = action_cache_metadata(encoder.image_tokens)
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('metadata') != metadata:
            raise ValueError('Action embedding cache does not match the encoder/actions. Delete models/action-embeddings.pt and run mario-agent prepare to regenerate it.')
        return FrozenEncoder._validate(payload['embeddings'], len(ACTIONS))
    vectors = encoder.encode_actions().clone()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'embeddings': vectors}, path)
    return vectors
