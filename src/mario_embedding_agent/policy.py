"""A direct trainable action projection and independent nonlinear value head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .actions import ACTIONS


class ActorCritic(nn.Module):
    def __init__(self, action_embeddings, temperature=0.2, left_action_logit_bias=-3.0):
        super().__init__()
        vectors = torch.as_tensor(action_embeddings, dtype=torch.float32).detach().clone()
        if vectors.ndim != 2 or vectors.shape[1] != 768 or not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
            raise ValueError('Expected finite nonzero action vectors with shape [actions,768].')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive.')
        vectors = F.normalize(vectors, dim=1)
        self.register_buffer('action_embeddings', vectors.clone())
        self.temperature = temperature
        if not math.isfinite(left_action_logit_bias):
            raise ValueError('Left-action logit bias must be finite.')
        self.left_action_logit_bias = float(left_action_logit_bias)
        self.register_buffer(
            'left_action_mask',
            torch.tensor([any(button == 'left' for button in action.buttons) for action in ACTIONS],
                         dtype=torch.float32),
            persistent=False,
        )
        self.actor = nn.Linear(768, len(vectors))
        with torch.no_grad():
            self.actor.weight.copy_(vectors / temperature)
            self.actor.bias.zero_()
        self.critic = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                                    nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        # A separate delta branch learns whether frame-to-frame visual motion
        # should change the action/value estimate. Zero output preserves the
        # previous single-frame policy until PPO learns a temporal residual.
        self.temporal_actor = nn.Linear(768, len(vectors), bias=False)
        nn.init.zeros_(self.temporal_actor.weight)
        self.temporal_critic = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(),
                                             nn.Linear(128, 1))
        nn.init.zeros_(self.temporal_critic[-1].weight)
        nn.init.zeros_(self.temporal_critic[-1].bias)

    def forward(self, features, action_bias=None, *, previous_features=None):
        features = F.normalize(features.to(torch.float32), dim=-1)
        if previous_features is None:
            temporal_delta = torch.zeros_like(features)
        else:
            previous_features = torch.as_tensor(previous_features, dtype=torch.float32, device=features.device)
            if previous_features.shape != features.shape:
                raise ValueError('Previous-frame embeddings must have the same shape as current embeddings.')
            temporal_delta = features - F.normalize(previous_features, dim=-1)
        logits = (self.actor(features) + self.temporal_actor(temporal_delta) +
                  self.left_action_logit_bias * self.left_action_mask)
        if action_bias is not None:
            action_bias = torch.as_tensor(action_bias, dtype=logits.dtype, device=logits.device)
            if logits.ndim == 1:
                if action_bias.ndim == 2 and action_bias.shape[0] == 1:
                    action_bias = action_bias[0]
                if action_bias.shape != logits.shape:
                    raise ValueError('Action bias must have one value per action.')
            else:
                if action_bias.ndim == 1:
                    action_bias = action_bias.unsqueeze(0)
                if action_bias.ndim != 2 or action_bias.shape[1] != logits.shape[1] or action_bias.shape[0] not in (1, logits.shape[0]):
                    raise ValueError('Batched action bias must have shape [actions] or [batch, actions].')
            logits = logits + action_bias
        values = self.critic(features).squeeze(-1) + self.temporal_critic(temporal_delta).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(self, features, sample=True, action_bias=None, previous_features=None):
        logits, value = self(features, action_bias=action_bias, previous_features=previous_features)
        distribution = Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(dim=-1)
        return int(action.item()), float(distribution.log_prob(action)), float(value), distribution.probs.tolist()
