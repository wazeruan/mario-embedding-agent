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
    duration_frames: int = 20


def gae(transitions, bootstrap_value, gamma=0.99, gae_lambda=0.95, reference_frames=20):
    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    estimate = 0.0
    next_value = float(bootstrap_value)
    for i in reversed(range(len(transitions))):
        item = transitions[i]
        if item.duration_frames <= 0 or reference_frames <= 0:
            raise ValueError('Transition duration and reference frames must be positive.')
        elapsed = item.duration_frames / reference_frames
        discount = gamma ** elapsed
        trace_decay = gae_lambda ** elapsed
        continuing = 0.0 if item.done else 1.0
        delta = item.reward + discount * continuing * next_value - item.value
        estimate = delta + discount * trace_decay * continuing * estimate
        advantages[i] = estimate
        next_value = item.value
    returns = advantages + torch.tensor([item.value for item in transitions])
    return advantages, returns


def update(policy, optimizer, transitions, bootstrap_value, *, epochs=2, minibatch_size=8,
           gamma=0.99, gae_lambda=0.95, clip=0.2, value_coef=0.5, entropy_coef=0.01,
           target_kl=0.02, reference_frames=20, value_clip=None):
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
    actor_mask = torch.tensor([item.action_bias is None for item in transitions], dtype=torch.bool)
    advantages, returns = gae(transitions, bootstrap_value, gamma, gae_lambda, reference_frames)
    if actor_mask.any():
        eligible = advantages[actor_mask]
        advantages = (advantages - eligible.mean()) / eligible.std(unbiased=False).clamp_min(1e-8)
    else:
        advantages.zero_()
    metrics = {key: [] for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
    stopped_early = False
    optimizer_steps = 0
    for _ in range(epochs):
        for indices in torch.randperm(len(transitions)).split(minibatch_size):
            logits, values = policy(features[indices], action_bias=action_biases[indices],
                                    previous_features=previous_features[indices])
            dist = Categorical(logits=logits)
            logs = dist.log_prob(actions[indices])
            log_ratio = logs - old_logs[indices]
            ratio = log_ratio.exp()
            eligible = actor_mask[indices]
            approx_kl = ((ratio[eligible]-1)-log_ratio[eligible]).mean() if eligible.any() else logits.sum()*0
            if target_kl is not None and float(approx_kl.detach()) > 1.5 * target_kl:
                stopped_early = True
                break
            surrogate = torch.minimum(ratio * advantages[indices],
                                      ratio.clamp(1-clip, 1+clip) * advantages[indices])
            actor_loss = -surrogate[eligible].mean() if eligible.any() else logits.sum()*0
            critic_loss = (values - returns[indices]).square()
            if value_clip is not None:
                # Optional PPO2-style value clipping; it slows critic fitting and
                # is off by default (Andrychowicz et al., 2021).
                clipped_values = old_values[indices] + (values - old_values[indices]).clamp(-value_clip, value_clip)
                critic_loss = torch.maximum(critic_loss, (clipped_values - returns[indices]).square())
            critic_loss = 0.5 * critic_loss.mean()
            entropy = dist.entropy()[eligible].mean() if eligible.any() else logits.sum()*0
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise ValueError('PPO loss is not finite.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not eligible.any():
                # Zero gradients still advance Adam momentum and weight decay.
                # Assisted batches must leave all actor branches untouched.
                for module in (policy.actor, policy.temporal_actor, getattr(policy, 'nonlinear_actor', None)):
                    if module is None:
                        continue
                    for parameter in module.parameters():
                        parameter.grad = None
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            optimizer_steps += 1
            for key, value in [('policy_loss', actor_loss), ('value_loss', critic_loss), ('entropy', entropy),
                               ('approx_kl', approx_kl),
                               ('clip_fraction', ((ratio[eligible]-1).abs() > clip).float().mean()
                                if eligible.any() else logits.sum()*0)]:
                metrics[key].append(float(value.detach()))
        if stopped_early:
            break
    result = {key: sum(values)/len(values) if values else 0.0 for key, values in metrics.items()}
    result.update(optimizer_steps=optimizer_steps, stopped_early=stopped_early,
                  actor_samples=int(actor_mask.sum()), recovery_samples=int((~actor_mask).sum()))
    with torch.no_grad():
        final_logits, predictions = policy(features, action_bias=action_biases, previous_features=previous_features)
        final_log_ratio = Categorical(logits=final_logits).log_prob(actions)-old_logs
        result['final_kl'] = float(((final_log_ratio[actor_mask].exp()-1)-final_log_ratio[actor_mask]).mean()) if actor_mask.any() else 0.0
        # Policy entropy on actor samples after the update, unaffected by minibatches
        # without actor samples or an early KL stop.
        result['actor_entropy'] = float(Categorical(logits=final_logits).entropy()[actor_mask].mean()) if actor_mask.any() else None
        variance = returns.var(unbiased=False)
        result['explained_variance'] = float(1-(returns-predictions).var(unbiased=False)/variance) if variance > 1e-12 else None
    result['mean_learning_reward'] = sum(item.reward for item in transitions)/len(transitions)
    return result
