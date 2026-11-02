import torch

from mario_embedding_agent.environment import make_env, step
from mario_embedding_agent.handwired import HandWiredPolicy
from mario_embedding_agent.rewards import StageDetector


def test_handwired_policy_has_no_trainable_parameters():
    policy = HandWiredPolicy()
    assert list(policy.parameters()) == []
    assert len(list(policy.buffers())) > 0


def test_handwired_policy_clears_world_1_1():
    policy, env = HandWiredPolicy(), make_env()
    frame, _ = env.reset(seed=0)
    detector = StageDetector()
    detector.observe(frame)
    state, cleared = None, False
    with torch.no_grad():
        for _ in range(4000):
            press, state, _ = policy(frame, state)
            frame, died = step(env, ('right', 'B', 'A') if press > 0.5 else ('right', 'B'))
            assert not died
            if detector.observe(frame):
                cleared = True
                break
    assert cleared
