"""Prepare, train and play the screenshot-only Mario policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import torch
from PIL import Image
from .actions import ACTIONS
from . import checkpoint
from .encoder import FrozenEncoder, download_model, prepare_actions
from .environment import make_env, step, COMMANDS
from .policy import ActorCritic
from .ppo import Transition, update
from .rewards import StageDetector, GapDetector, RightwardProgress, reward
from .display import Display

ROOT = Path(__file__).resolve().parents[2]


def reset_critic_value_scale(policy, optimizer):
    """Keep learned critic features but clear value output and stale Adam moments."""
    with torch.no_grad():
        policy.critic[-1].weight.zero_()
        policy.critic[-1].bias.zero_()
    for parameter in policy.critic.parameters():
        optimizer.state.pop(parameter, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'play', 'evaluate'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--mode', choices=['sync', 'realtime'],
                        help='Emulator pacing; defaults to sync for evaluation and realtime otherwise')
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--rollout', type=int, default=16)
    parser.add_argument('--total-frames', type=int, help='Explicit bounded diagnostic only; otherwise unbounded')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/latest.pt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10,
                        help='Death-ended evaluation attempts before stopping')
    args = parser.parse_args()
    args.mode = args.mode or ('sync' if args.command == 'evaluate' else 'realtime')
    if args.fps <= 0 or args.rollout < 2 or (args.total_frames is not None and args.total_frames <= 0):
        parser.error('fps must be positive, rollout >= 2, total-frames positive')
    if args.command == 'evaluate' and (args.episodes <= 0 or args.total_frames is not None):
        parser.error('evaluate requires positive --episodes and stops only at death, not by frame count')
    directory = ROOT/'models/embeddinggemma-2'
    if args.command == 'prepare':
        download_model(directory)
    encoder = FrozenEncoder(directory, device=args.device)
    embeddings = prepare_actions(encoder, ROOT/'models/action-embeddings.pt')
    if args.command == 'prepare':
        print(f'Prepared {len(ACTIONS)} actions: {tuple(embeddings.shape)}')
        return
    payload = checkpoint.load(args.resume or args.checkpoint) if args.resume or args.command in ('play', 'evaluate') else None
    config = {'reward_scale': 50.0, 'image_tokens': 70, 'seed': args.seed,
              'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
              'gap_bonus': 1.0, 'gap_detector': 'overworld-rgb-v1',
              'death_penalty': 50.0, 'pixels_per_reward': 8.0,
              'progress_clip': 5.0, 'reward_version': 'signed-progress-v2',
              'left_action_logit_bias': -3.0, 'entropy_coef': 0.001,
              'stagnation_threshold': 4, 'stagnation_deadband_px': 1.0,
              'recovery_actions': ['down', 'run_left', 'run_right_jump_long', 'right_jump_short'],
              'recovery_logit_bias': 8.0}
    reward_objective_migrated = False
    policy_behavior_migrated = False
    recovery_behavior_migrated = False
    if payload:
        # Preserve learned weights and optimizer while replacing old reward settings.
        saved_config = dict(payload['config'])
        old_reward_version = saved_config.get('reward_version')
        old_left_bias = saved_config.get('left_action_logit_bias')
        old_entropy_coef = saved_config.get('entropy_coef')
        recovery_keys = ('stagnation_threshold', 'stagnation_deadband_px',
                         'recovery_actions', 'recovery_logit_bias')
        recovery_behavior_migrated = any(saved_config.get(key) != config[key] for key in recovery_keys)
        for key in ('movement_reward', 'motion_detector'):
            saved_config.pop(key, None)
        saved_config.update({key: config[key] for key in (
            'reward_scale', 'death_penalty', 'pixels_per_reward', 'progress_clip', 'reward_version',
            'left_action_logit_bias', 'entropy_coef', 'stagnation_threshold',
            'stagnation_deadband_px', 'recovery_actions', 'recovery_logit_bias')})
        saved_config.setdefault('gap_bonus', 1.0)
        saved_config.setdefault('gap_detector', 'overworld-rgb-v1')
        reward_objective_migrated = old_reward_version != config['reward_version']
        policy_behavior_migrated = (old_left_bias != config['left_action_logit_bias'] or
                                    old_entropy_coef != config['entropy_coef'])
        if {k: v for k, v in saved_config.items() if k != 'seed'} != {k: v for k, v in config.items() if k != 'seed'}:
            raise ValueError('Checkpoint training configuration is incompatible')
        config = saved_config
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    policy = ActorCritic(embeddings, left_action_logit_bias=config['left_action_logit_bias'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config['learning_rate'])
    state = {'frames': 0, 'decisions': 0, 'updates': 0, 'deaths': 0, 'clears': 0,
             'episode_seed': config['seed'], 'episode_commands': [], 'gaps': 0,
             'stagnant_decisions': 0}
    if args.command == 'evaluate':
        state['episode_seed'] = args.seed
    if payload:
        policy.load_state_dict(payload['policy'])
        if args.command == 'evaluate':
            # Evaluate fresh, death-ended attempts under an explicit seed sequence.
            state['episode_commands'] = []
        else:
            optimizer.load_state_dict(payload['optimizer'])
            if args.command == 'train' and reward_objective_migrated:
                reset_critic_value_scale(policy, optimizer)
            state.update({k: v for k, v in payload['training_state'].items() if k != 'torch_rng_state'})
            torch.set_rng_state(payload['training_state']['torch_rng_state'])
    env = make_env()
    frame, _ = env.reset(seed=state['episode_seed'])
    detector = StageDetector()
    gap_detector = GapDetector()
    motion = RightwardProgress()
    motion.reset(frame)
    detector.observe(frame)
    gap_detector.observe(frame)
    # Deterministic replay reconstructs the current episode without reading RAM.
    for index in state['episode_commands']:
        frame, died = step(env, COMMANDS[index])
        if detector.observe(frame):
            gap_detector = GapDetector()
            motion.reset(frame)
        else:
            gap_detector.observe(frame)
        motion.observe(frame)
        if died:
            raise ValueError('Saved episode replay unexpectedly terminated')
    display = None if args.headless else Display()
    logdir = ROOT/'runs'
    logdir.mkdir(exist_ok=True)
    log = (logdir/f'{args.command}.jsonl').open('a')
    worker = ThreadPoolExecutor(max_workers=1)
    future = None
    generation = 0
    active = None
    rollout = []
    metrics = {}
    save_due = False
    policy_frame = None
    probabilities = []
    last_reward = 0.0
    inference_ms = 0.0
    evaluation = {'episodes': 0, 'frames': [], 'decisions': [], 'raw_rewards': [],
                  'rightward_progress_px': [], 'gap_crossings': [], 'world_transitions': [],
                  'episode_start_frames': state['frames'],
                  'episode_start_decisions': state['decisions'],
                  'episode_start_gaps': state['gaps'], 'episode_start_clears': state['clears'],
                  'episode_raw_reward': 0.0, 'episode_progress_px': 0.0}
    stop_evaluation = False
    started = time.perf_counter()
    start_frames, start_decisions = state['frames'], state['decisions']
    next_tick = started
    last_report = started

    def emit(kind, **data):
        log.write(json.dumps({'event': kind, **state_counts(), **data})+'\n')
        log.flush()

    def state_counts():
        return {k: state[k] for k in ('frames', 'decisions', 'updates', 'deaths', 'clears', 'gaps',
                                      'stagnant_decisions')}

    def finish(raw, done, bootstrap=0.0, outcome='step'):
        nonlocal active, last_reward, metrics, save_due
        last_reward = raw
        if outcome in ('death', 'level_clear'):
            state['stagnant_decisions'] = 0
        if args.command == 'evaluate':
            # Count terminal rewards even if the death/transition lands while
            # the controller has no active policy decision to credit.
            evaluation['episode_raw_reward'] += raw
            if active is not None:
                evaluation['episode_progress_px'] += active['right_progress']
        if active is not None:
            if outcome not in ('death', 'level_clear') and (
                  active['right_progress'] <= config['stagnation_deadband_px'] and
                  active['gap_bonus'] <= 0):
                state['stagnant_decisions'] += 1
            elif outcome not in ('death', 'level_clear'):
                state['stagnant_decisions'] = 0
            emit('decision_result', decision=state['decisions'], action=ACTIONS[active['action']].name,
                 duration_frames=max(0, state['frames']-active['observed']),
                 displacement_px=round(active['right_progress'], 3), gap_bonus=active['gap_bonus'],
                 raw_reward=raw, scaled_reward=raw/config['reward_scale'], outcome=outcome,
                 recovery_action=active['recovery_action'], stagnant_decisions=state['stagnant_decisions'])
            if args.command == 'train':
                rollout.append(Transition(active['features'], active['action'], active['log_prob'],
                                          active['value'], raw/config['reward_scale'], done,
                                          active['action_bias']))
        active = None
        if len(rollout) >= args.rollout:
            metrics = update(policy, optimizer, rollout, bootstrap,
                             entropy_coef=config['entropy_coef'])
            rollout.clear()
            state['updates'] += 1
            save_due = True
            emit('ppo', **metrics)

    def encode(image):
        before = time.perf_counter()
        features = encoder.encode_frame(image)
        return features, (time.perf_counter()-before)*1000

    emit('started', mode=args.mode, gap_bonus=config['gap_bonus'], reward_objective_migrated=reward_objective_migrated,
         critic_reset_for_reward_migration=bool(args.command == 'train' and payload and reward_objective_migrated),
         policy_behavior_migrated=policy_behavior_migrated,
         recovery_behavior_migrated=recovery_behavior_migrated,
         left_action_logit_bias=config['left_action_logit_bias'], entropy_coef=config['entropy_coef'],
         stagnation_threshold=config['stagnation_threshold'], recovery_actions=config['recovery_actions'],
         recovery_logit_bias=config['recovery_logit_bias'])
    print(f'{args.command}: {len(ACTIONS)} actions; mode={args.mode}; PPO rollout={args.rollout}; encoder={encoder.device}', flush=True)
    try:
        while (not stop_evaluation and
               (args.total_frames is None or state['frames']-start_frames < args.total_frames)):
            now = time.perf_counter()
            ready_for_decision = active is None or active['position'] >= len(active['commands'])
            if ready_for_decision and future is None:
                captured = frame.copy()
                submitted = (generation, state['frames'], captured)
                if args.mode == 'sync':
                    features, inference_ms = encode(captured)
                    future = None
                    ready = (features, submitted)
                else:
                    future = worker.submit(encode, captured)
                    ready = None
            else:
                ready = None
            if future is not None and future.done():
                features, inference_ms = future.result()
                future = None
                ready = (features, submitted) if submitted[0] == generation else None
            if ready is not None:
                features, observation = ready
                with torch.no_grad():
                    bootstrap = float(policy(features)[1])
                if active is not None:
                    finish(reward(False, False, active['right_progress'], active['gap_bonus'],
                                  death_penalty=config['death_penalty'],
                                  pixels_per_reward=config['pixels_per_reward'],
                                  progress_clip=config['progress_clip']), False, bootstrap)
                recovery_action = None
                action_bias = None
                if state['stagnant_decisions'] >= config['stagnation_threshold']:
                    index = ((state['stagnant_decisions']-config['stagnation_threshold']) %
                             len(config['recovery_actions']))
                    recovery_action = config['recovery_actions'][index]
                    action_bias = torch.zeros(len(ACTIONS))
                    recovery_index = next(i for i, item in enumerate(ACTIONS) if item.name == recovery_action)
                    action_bias[recovery_index] = config['recovery_logit_bias']
                action, log_prob, value, probabilities = policy.act(
                    features, sample=args.command == 'train', action_bias=action_bias)
                active = dict(features=features, action=action, log_prob=log_prob, value=value,
                              commands=ACTIONS[action].commands(), position=0, observed=observation[1], gap_bonus=0.0,
                              right_progress=0.0, recovery_action=recovery_action, action_bias=action_bias)
                policy_frame = observation[2]
                state['decisions'] += 1
                emit('decision', action=ACTIONS[action].name, inference_ms=inference_ms,
                     observation_age_frames=state['frames']-observation[1], value=value,
                     recovery_action=recovery_action, stagnant_decisions=state['stagnant_decisions'])
            command = ()
            if active is not None:
                position = active['position']
                command = active['commands'][min(position, len(active['commands'])-1)]
                active['position'] += 1
            frame, died = step(env, command)
            state['frames'] += 1
            state['episode_commands'].append(COMMANDS.index(command))
            right_progress = motion.observe(frame)
            if right_progress is not None and active is not None:
                active['right_progress'] += right_progress
            cleared = detector.observe(frame)
            crossed = gap_detector.observe(frame)
            if crossed and not died and not cleared and active is not None:
                active['gap_bonus'] += config['gap_bonus']
                state['gaps'] += 1
                image_path = logdir/f'gap-{state["frames"]}.png'
                Image.fromarray(frame).save(image_path)
                emit('gap_crossing', bonus=config['gap_bonus'], captured_frame=str(image_path))
            if died or cleared:
                outcome = 'death' if died else 'clear'
                image_path = logdir/f'{outcome}-{state["frames"]}.png'
                Image.fromarray(frame).save(image_path)
                finish(reward(died, cleared, death_penalty=config['death_penalty']), True,
                       outcome='death' if died else 'level_clear')
                state['deaths' if died else 'clears'] += 1
                emit(outcome, raw_reward=last_reward, captured_frame=str(image_path))
                generation += 1
                gap_detector = GapDetector()
                if cleared:
                    motion.reset(frame)
                if died:
                    if args.command == 'evaluate':
                        evaluation['episodes'] += 1
                        episode_frames = state['frames']-evaluation['episode_start_frames']
                        episode_decisions = state['decisions']-evaluation['episode_start_decisions']
                        episode_gaps = state['gaps']-evaluation['episode_start_gaps']
                        episode_clears = state['clears']-evaluation['episode_start_clears']
                        evaluation['frames'].append(episode_frames)
                        evaluation['decisions'].append(episode_decisions)
                        evaluation['raw_rewards'].append(evaluation['episode_raw_reward'])
                        evaluation['rightward_progress_px'].append(evaluation['episode_progress_px'])
                        evaluation['gap_crossings'].append(episode_gaps)
                        evaluation['world_transitions'].append(episode_clears)
                        emit('evaluation_episode', episode=evaluation['episodes'], seed=state['episode_seed'],
                             frames=episode_frames, decisions=episode_decisions,
                             raw_reward=evaluation['episode_raw_reward'],
                             rightward_progress_px=evaluation['episode_progress_px'],
                             gap_crossings=episode_gaps, world_transitions=episode_clears,
                             died=True)
                        if evaluation['episodes'] >= args.episodes:
                            stop_evaluation = True
                        else:
                            state['episode_seed'] += 1
                            state['episode_commands'] = []
                            frame, _ = env.reset(seed=state['episode_seed'])
                            motion.reset(frame)
                            detector = StageDetector()
                            detector.observe(frame)
                            evaluation['episode_start_frames'] = state['frames']
                            evaluation['episode_start_decisions'] = state['decisions']
                            evaluation['episode_start_gaps'] = state['gaps']
                            evaluation['episode_start_clears'] = state['clears']
                            evaluation['episode_raw_reward'] = 0.0
                            evaluation['episode_progress_px'] = 0.0
                    else:
                        state['episode_seed'] += 1
                        state['episode_commands'] = []
                        frame, _ = env.reset(seed=state['episode_seed'])
                        motion.reset(frame)
                        detector = StageDetector()
                        detector.observe(frame)
            if stop_evaluation:
                break
            if args.command == 'train' and save_due:
                checkpoint.save(args.checkpoint, policy, optimizer, config, state)
                save_due = False
            elapsed = max(time.perf_counter()-started, .001)
            fps = (state['frames']-start_frames)/elapsed
            dps = (state['decisions']-start_decisions)/elapsed
            if display:
                recovery = active['recovery_action'] if active and active['recovery_action'] else 'off'
                lines = [f'{args.command} / {args.mode}', f'FPS {fps:.1f}  decisions/s {dps:.2f}',
                         f'Frame {state["frames"]}', f'Action: {ACTIONS[active["action"]].name if active else "waiting"}',
                         f'Encoder {inference_ms:.1f} ms', f'Input age {state["frames"]-active["observed"] if active else 0} frames',
                         f'Stagnant decisions {state["stagnant_decisions"]}  recovery {recovery}',
                         f'Rollout {len(rollout)}/{args.rollout}', f'PPO updates {state["updates"]}',
                         f'Dead {state["deaths"]} clear {state["clears"]} gaps {state["gaps"]}', f'Reward {last_reward:g} / {last_reward/config["reward_scale"]:g}',
                         *[f'{k}: {metrics[k]:.4g}' for k in ('policy_loss', 'value_loss', 'entropy', 'explained_variance') if isinstance(metrics.get(k), (float, int))],
                         'Top action probabilities:',
                         *[f'{ACTIONS[i].name}: {probabilities[i]:.2f}' for i in sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]],
                         'Below: policy input frame']
                if not display.draw(frame, policy_frame, lines):
                    break
            if time.perf_counter()-last_report >= 2:
                print(f'frames={state["frames"]} fps={fps:.1f} decisions/s={dps:.2f} updates={state["updates"]} deaths={state["deaths"]} reward={last_reward:g}', flush=True)
                last_report = time.perf_counter()
            next_tick = max(next_tick + 1/args.fps, time.perf_counter())
            time.sleep(max(0, next_tick-time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        worker.shutdown(wait=True, cancel_futures=True)
        if args.command == 'train':
            checkpoint.save(args.checkpoint, policy, optimizer, config, state)
        if args.command == 'evaluate':
            completed = evaluation['episodes']
            emit('evaluation_complete', requested_episodes=args.episodes, completed_episodes=completed,
                 complete=completed == args.episodes,
                 mean_frames=(sum(evaluation['frames'])/completed if completed else None),
                 mean_decisions=(sum(evaluation['decisions'])/completed if completed else None),
                 mean_raw_reward=(sum(evaluation['raw_rewards'])/completed if completed else None),
                 mean_rightward_progress_px=(sum(evaluation['rightward_progress_px'])/completed if completed else None),
                 total_gap_crossings=sum(evaluation['gap_crossings']),
                 total_world_transitions=sum(evaluation['world_transitions']))
        emit('stopped', partial_rollout_discarded=len(rollout))
        log.close()
        env.close()
        if display:
            display.close()


if __name__ == '__main__':
    main()
