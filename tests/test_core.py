import math
import json
import sys
import types

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mario_embedding_agent.actions import ACTIONS, action_manifest
from mario_embedding_agent.policy import ActorCritic
from mario_embedding_agent.ppo import Transition, gae, update
from mario_embedding_agent import checkpoint
from mario_embedding_agent.encoder import FrozenEncoder, MODEL_ID, MODEL_REVISION, prepare_actions


def vectors():
    return torch.randn(len(ACTIONS), 768, generator=torch.Generator().manual_seed(42))


def test_macro_jump_duration_and_release_preserve_direction():
    assert len(ACTIONS) == 18
    assert len({a.name for a in ACTIONS}) == 18
    for action in ACTIONS:
        commands = action.commands()
        assert len(commands) == action.hold_frames + action.release_frames
        if 'A' in action.buttons:
            assert action.hold_frames == (4 if action.name.endswith('short') else 20)
            assert commands[:action.hold_frames] == (action.buttons,) * action.hold_frames
            assert commands[-2:] == (tuple(b for b in action.buttons if b != 'A'),) * 2
        elif action.name == 'fire':
            assert commands[-2:] == ((), ())
    assert ACTIONS[0].commands() == ((),) * 4


def test_actor_initializes_cosine_logits_but_learns_independently():
    original = vectors()
    expected = F.normalize(original.clone(), dim=1)
    policy = ActorCritic(original, temperature=.2)
    original.zero_()
    features = torch.randn(3, 768)
    logits, values = policy(features)
    torch.testing.assert_close(logits, F.normalize(features, dim=1) @ expected.T / .2 - 3 * policy.left_action_mask)
    assert values.shape == (3,)
    assert not policy.action_embeddings.requires_grad
    assert policy.actor.weight.data_ptr() != policy.action_embeddings.data_ptr()
    torch.testing.assert_close(values, torch.zeros(3))


def test_gae_death_cuts_bootstrap_and_next_episode_rewards():
    f = torch.zeros(768)
    rollout = [Transition(f, 0, 0, .2, -.001, False),
               Transition(f, 0, 0, .3, -1, True),
               Transition(f, 0, 0, .4, .001, False)]
    adv, returns = gae(rollout, 10, gamma=.9, gae_lambda=.8)
    expected_last = .001 + .9 * 10 - .4
    expected_death = -1 - .3
    expected_first = -.001 + .9*.3 - .2 + .9*.8*expected_death
    torch.testing.assert_close(adv, torch.tensor([expected_first, expected_death, expected_last]))
    torch.testing.assert_close(returns[1], torch.tensor(-1.))


def test_ppo_updates_actor_and_critic_with_frozen_features_only():
    torch.manual_seed(123)
    policy = ActorCritic(vectors())
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    rollout = []
    for i in range(16):
        features = F.normalize(torch.randn(768), dim=0)
        action, log_prob, value, _ = policy.act(features)
        rollout.append(Transition(features, action, log_prob, value, -1. if i == 15 else -.001, i == 15))
    before_actor = policy.actor.weight.detach().clone()
    before_critic = policy.critic[-1].weight.detach().clone()
    before_actions = policy.action_embeddings.clone()
    before_features = [t.features.clone() for t in rollout]
    metrics = update(policy, optimizer, rollout, 0, epochs=2, minibatch_size=8)
    assert not torch.equal(policy.actor.weight, before_actor)
    assert not torch.equal(policy.critic[-1].weight, before_critic)
    torch.testing.assert_close(policy.action_embeddings, before_actions)
    for before, item in zip(before_features, rollout):
        torch.testing.assert_close(before, item.features)
    assert all(v is None or math.isfinite(v) for v in metrics.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    assert all(int(s['step']) == 4 for s in optimizer.state.values())


def test_checkpoint_roundtrip_and_incompatible_manifest(tmp_path):
    policy = ActorCritic(vectors())
    optimizer = torch.optim.Adam(policy.parameters())
    path = tmp_path/'checkpoint.pt'
    checkpoint.save(path, policy, optimizer, {'rollout_steps': 16}, {'updates': 3})
    payload = checkpoint.load(path)
    assert payload['training_state']['updates'] == 3
    assert payload['actions'] == action_manifest()
    restored = ActorCritic(vectors())
    restored.load_state_dict(payload['policy'])
    for a, b in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    payload['actions'][0]['hold_frames'] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='architecture'):
        checkpoint.load(path)


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def encode(self, inputs, **kwargs):
            self.calls.append((inputs, kwargs))
            return torch.arange(1, 769, dtype=torch.float32).repeat(len(inputs), 1)

    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeModel))
    (tmp_path/'mario-model.json').write_text(json.dumps({'model_id': MODEL_ID, 'revision': MODEL_REVISION}))
    return FrozenEncoder(tmp_path, device='cpu', cache_size=2)


def test_frame_features_normal_autograd_safe_bounded_cache(encoder):
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    feature = encoder.encode_frame(frame)
    assert not torch.is_inference(feature)
    policy = ActorCritic(vectors())
    policy(feature)[0].sum().backward()
    cached = encoder.encode_frame(frame)
    torch.testing.assert_close(feature, cached)
    cached.zero_()
    assert encoder.encode_frame(frame).norm() > 0
    assert len(encoder.model.calls) == 1
    for pixel in (1, 2):
        changed = frame.copy()
        changed[0, 0, 0] = pixel
        encoder.encode_frame(changed)
    assert len(encoder.cache) == 2
    encoder.encode_frame(frame)
    assert len(encoder.model.calls) == 4
    assert encoder.model.calls[0][1]['processing_kwargs']['image']['max_soft_tokens'] == 70


def test_preencoded_action_cache_reused_and_metadata_checked(encoder, tmp_path):
    path = tmp_path/'actions.pt'
    first = prepare_actions(encoder, path)
    second = prepare_actions(encoder, path)
    assert len(encoder.model.calls) == 1
    torch.testing.assert_close(first, second)
    assert len(encoder.model.calls[0][0]) == 18
    payload = torch.load(path, weights_only=True)
    payload['metadata']['revision'] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='cache'):
        prepare_actions(encoder, path)


def test_uniform_actor_bias_downweights_every_left_action_without_masking():
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    feature = torch.ones(768)
    logits, _ = policy(feature)
    left = torch.tensor(['left' in action.buttons for action in ACTIONS])
    assert left.sum() == 6
    torch.testing.assert_close(logits[left], torch.full((6,), 2.))
    torch.testing.assert_close(logits[~left], torch.full((12,), 5.))
    _, _, _, probabilities = policy.act(feature)
    probs = torch.tensor(probabilities)
    assert 0. < float(probs[left].sum()) < .025
    torch.testing.assert_close(probs[left][0]/probs[~left][0], torch.tensor(math.exp(-3)))
    torch.manual_seed(789)
    chosen = [policy.act(feature)[0] for _ in range(500)]
    assert any(left[index] for index in chosen)
    assert 'left_action_mask' not in policy.state_dict()  # Old checkpoints load strictly.


def test_ppo_logprobs_and_entropy_use_same_left_bias_as_sampling():
    policy = ActorCritic(torch.ones(len(ACTIONS), 768))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.)
    feature = torch.ones(768)
    torch.manual_seed(88)
    rollout = []
    for i in range(8):
        action, log_prob, value, probabilities = policy.act(feature)
        rollout.append(Transition(feature, action, log_prob, value, .1, i == 7))
    distribution = torch.distributions.Categorical(probs=torch.tensor(probabilities))
    metrics = update(policy, optimizer, rollout, 0., epochs=1, minibatch_size=8, entropy_coef=.001)
    assert metrics['approx_kl'] == pytest.approx(0., abs=1e-7)
    assert metrics['entropy'] == pytest.approx(float(distribution.entropy()), abs=1e-6)
    assert metrics['clip_fraction'] == 0.


@pytest.mark.parametrize('bias', [float('nan'), float('inf'), -float('inf')])
def test_left_action_bias_requires_finite_value(bias):
    with pytest.raises(ValueError, match='finite'):
        ActorCritic(vectors(), left_action_logit_bias=bias)
