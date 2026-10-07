"""A direct trainable action projection and independent nonlinear value head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, action_embeddings, temperature=0.2):
        super().__init__()
        vectors = torch.as_tensor(action_embeddings, dtype=torch.float32).detach().clone()
        if vectors.ndim != 2 or vectors.shape[1] != 768 or not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
            raise ValueError('Expected finite nonzero action vectors with shape [actions,768].')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive.')
        vectors = F.normalize(vectors, dim=1)
        self.register_buffer('action_embeddings', vectors.clone())
        self.temperature = temperature
        self.actor = nn.Linear(768, len(vectors))
        with torch.no_grad():
            self.actor.weight.copy_(vectors / temperature)
            self.actor.bias.zero_()
        self.critic = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                                    nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, features):
        features = F.normalize(features.to(torch.float32), dim=-1)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def act(self, features, sample=True):
        logits, value = self(features)
        distribution = Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(dim=-1)
        return int(action.item()), float(distribution.log_prob(action)), float(value), distribution.probs.tolist()
