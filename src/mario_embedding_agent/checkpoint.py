"""Atomic portable checkpoints; strict encoder/action/reward compatibility."""
from pathlib import Path
import os
import torch

from .actions import action_manifest
from .encoder import MODEL_ID, MODEL_REVISION, QUESTION

FORMAT = 'mario-direct-actor-ppo-v1'


def save(path, policy, optimizer, config, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': FORMAT, 'model_id': MODEL_ID, 'revision': MODEL_REVISION,
               'question': QUESTION, 'actions': action_manifest(), 'config': config,
               'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
               'training_state': {**state, 'torch_rng_state': torch.get_rng_state()}}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format') != FORMAT or payload.get('model_id') != MODEL_ID or
            payload.get('revision') != MODEL_REVISION or payload.get('question') != QUESTION or
            payload.get('actions') != action_manifest()):
        raise ValueError('Checkpoint does not match this rewritten actor/action/encoder architecture.')
    return payload
