"""On-policy clipped PPO on cached frozen screenshot features."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.distributions import Categorical


@dataclass
class Transition:
    features: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_bias: torch.Tensor | None = None
    previous_features: torch.Tensor | None = None


def gae(transitions, bootstrap_value, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    estimate = 0.0
    next_value = float(bootstrap_value)
    for i in reversed(range(len(transitions))):
        item = transitions[i]
        continuing = 0.0 if item.done else 1.0
        delta = item.reward + gamma * continuing * next_value - item.value
        estimate = delta + gamma * gae_lambda * continuing * estimate
        advantages[i] = estimate
        next_value = item.value
    returns = advantages + torch.tensor([item.value for item in transitions])
    return advantages, returns


def update(policy, optimizer, transitions, bootstrap_value, *, epochs=2, minibatch_size=8,
           gamma=0.99, gae_lambda=0.95, clip=0.2, value_coef=0.5, entropy_coef=0.01):
    if not transitions:
        raise ValueError('Cannot update an empty rollout.')
    features = torch.stack([item.features.detach().clone() for item in transitions])
    previous_features = torch.stack([
        item.features.detach().clone() if item.previous_features is None
        else item.previous_features.detach().clone()
        for item in transitions
    ])
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long)
    action_biases = torch.stack([
        torch.zeros(policy.actor.out_features) if item.action_bias is None
        else torch.as_tensor(item.action_bias, dtype=torch.float32).detach().clone()
        for item in transitions
    ])
    old_logs = torch.tensor([item.log_prob for item in transitions])
    old_values = torch.tensor([item.value for item in transitions])
    advantages, returns = gae(transitions, bootstrap_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    metrics = {key: [] for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
    for _ in range(epochs):
        for indices in torch.randperm(len(transitions)).split(minibatch_size):
            logits, values = policy(features[indices], action_bias=action_biases[indices],
                                    previous_features=previous_features[indices])
            dist = Categorical(logits=logits)
            logs = dist.log_prob(actions[indices])
            log_ratio = logs - old_logs[indices]
            ratio = log_ratio.exp()
            actor_loss = -torch.minimum(ratio * advantages[indices],
                                       ratio.clamp(1-clip, 1+clip) * advantages[indices]).mean()
            clipped_values = old_values[indices] + (values - old_values[indices]).clamp(-clip, clip)
            critic_loss = 0.5 * torch.maximum((values - returns[indices]).square(),
                                             (clipped_values - returns[indices]).square()).mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise ValueError('PPO loss is not finite.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            for key, value in [('policy_loss', actor_loss), ('value_loss', critic_loss), ('entropy', entropy),
                               ('approx_kl', ((ratio-1)-log_ratio).mean()),
                               ('clip_fraction', ((ratio-1).abs() > clip).float().mean())]:
                metrics[key].append(float(value.detach()))
    result = {key: sum(values)/len(values) for key, values in metrics.items()}
    with torch.no_grad():
        predictions = policy(features, previous_features=previous_features)[1]
        variance = returns.var(unbiased=False)
        result['explained_variance'] = float(1-(returns-predictions).var(unbiased=False)/variance) if variance > 1e-12 else None
    result['mean_learning_reward'] = sum(item.reward for item in transitions)/len(transitions)
    return result
