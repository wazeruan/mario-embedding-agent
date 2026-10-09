import json
import sys

import numpy as np
import torch


def test_evaluate_stops_after_death_count_without_training_or_final_reset(monkeypatch, tmp_path):
    from mario_embedding_agent import cli

    class Encoder:
        device = 'cpu'

        def __init__(self, *args, **kwargs):
            pass

        def encode_frame(self, frame):
            return torch.ones(768)

    class Env:
        def __init__(self):
            self.resets = []
            self.episode_frames = 0

        def reset(self, seed):
            self.resets.append(seed)
            self.episode_frames = 0
            return np.zeros((240, 256, 3), dtype=np.uint8), {}

        def close(self):
            pass

    class NoTransitions:
        def observe(self, frame):
            return False

    class NoMotion:
        def reset(self, frame):
            pass

        def observe(self, frame):
            return None

    env = Env()
    policy = cli.ActorCritic(torch.ones(18, 768))
    config = {'reward_scale': 50.0, 'image_tokens': 70, 'seed': 0,
              'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
              'gap_bonus': 1.0, 'gap_detector': 'overworld-rgb-v1',
              'death_penalty': 50.0, 'pixels_per_reward': 8.0,
              'progress_clip': 5.0, 'reward_version': 'signed-progress-v2',
              'left_action_logit_bias': -3.0, 'entropy_coef': 0.001,
              'stagnation_threshold': 4, 'stagnation_deadband_px': 1.0,
              'recovery_actions': ['down', 'run_left', 'run_right_jump_long', 'right_jump_short'],
              'recovery_logit_bias': 8.0}
    payload = {'config': config, 'policy': policy.state_dict(), 'optimizer': {},
               'training_state': {'frames': 1000, 'decisions': 50, 'deaths': 9,
                                  'episode_seed': 99, 'episode_commands': [1]}}
    updates = []
    saves = []

    def fake_step(environment, command):
        environment.episode_frames += 1
        frame = np.full((240, 256, 3), environment.episode_frames, dtype=np.uint8)
        return frame, environment.episode_frames == 3

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'FrozenEncoder', Encoder)
    monkeypatch.setattr(cli, 'prepare_actions', lambda *args: torch.ones(18, 768))
    monkeypatch.setattr(cli, 'make_env', lambda: env)
    monkeypatch.setattr(cli, 'StageDetector', NoTransitions)
    monkeypatch.setattr(cli, 'GapDetector', NoTransitions)
    monkeypatch.setattr(cli, 'RightwardProgress', NoMotion)
    monkeypatch.setattr(cli, 'step', fake_step)
    monkeypatch.setattr(cli.checkpoint, 'load', lambda _: payload)
    monkeypatch.setattr(cli.checkpoint, 'save', lambda *args: saves.append(args))
    monkeypatch.setattr(cli, 'update', lambda *args, **kwargs: updates.append(args))
    monkeypatch.setattr(cli.ActorCritic, 'act', lambda *args, **kwargs: (0, 0.0, 0.0, [1.0] + [0.0] * 17))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['mario-agent', 'evaluate', '--headless',
                                    '--episodes', '2', '--seed', '7', '--checkpoint', 'policy.pt'])

    cli.main()

    # Seeds advance only at observed deaths; the final death does not trigger an unused reset.
    assert env.resets == [7, 8]
    assert updates == []
    assert saves == []
    events = [json.loads(line) for line in (tmp_path/'runs/evaluate.jsonl').read_text().splitlines()]
    assert next(event for event in events if event['event'] == 'started')['mode'] == 'sync'
    attempts = [event for event in events if event['event'] == 'evaluation_episode']
    assert [event['seed'] for event in attempts] == [7, 8]
    assert [event['frames'] for event in attempts] == [3, 3]
    assert [event['raw_reward'] for event in attempts] == [-15.0, -15.0]
    complete = next(event for event in events if event['event'] == 'evaluation_complete')
    assert complete['complete'] is True
    assert complete['completed_episodes'] == 2
    assert complete['mean_frames'] == 3
    assert complete['mean_raw_reward'] == -15
    assert len(list((tmp_path/'runs').glob('death-*.png'))) == 2

