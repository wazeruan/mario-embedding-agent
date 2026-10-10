"""Prepare, train and play the screenshot-only Mario policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from PIL import Image
from .actions import ACTIONS
from . import checkpoint
from .encoder import FrozenEncoder, download_model, prepare_actions
from .environment import make_env, step, COMMANDS
from .policy import ActorCritic, CALIBRATION_PRIOR_COUNT
from .ppo import Transition, update
from .rewards import StageDetector, GapDetector, RightwardProgress, ProgressFrontier, reward
from .display import Display
from .recovery import PipeExitRecovery

ROOT = Path(__file__).resolve().parents[2]


def reset_critic_value_scale(policy, optimizer):
    """Keep learned critic features but clear value output and stale Adam moments."""
    critics = [policy.critic, policy.temporal_critic]
    if hasattr(policy, 'joint_critic'):
        critics.append(policy.joint_critic)
    with torch.no_grad():
        for critic in critics:
            critic[-1].weight.zero_()
            critic[-1].bias.zero_()
    for critic in critics:
        for parameter in critic.parameters():
            optimizer.state.pop(parameter, None)


def load_policy_weights(policy, weights):
    """Allow a pre-temporal checkpoint to start with zeroed temporal adapters.

    MLP checkpoints from before running input statistics lack `feature_count`:
    calibrated statistics then count as a prior, identity statistics as none.
    """
    result = policy.load_state_dict(weights, strict=False)
    allowed_missing = {name for name in policy.state_dict()
                       if name.startswith(('temporal_actor.', 'temporal_critic.'))}
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    statistics_missing = 'feature_count' in missing
    missing.discard('feature_count')
    if (missing and missing != allowed_missing) or unexpected:
        raise ValueError(f'Checkpoint policy keys are incompatible: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    if statistics_missing:
        identity = bool((policy.feature_mean == 0).all() and (policy.feature_std == 1).all())
        policy.feature_count.fill_(0 if identity else CALIBRATION_PRIOR_COUNT)
    return bool(missing)


def load_optimizer_state(optimizer, saved_state, policy):
    """Extend legacy AdamW parameter IDs for newly added temporal adapters."""
    saved_state = deepcopy(saved_state)
    if len(saved_state['param_groups']) == len(optimizer.param_groups) == 1:
        group = saved_state['param_groups'][0]
        saved_ids = group['params']
        current_count = len(optimizer.param_groups[0]['params'])
        legacy_count = sum(not name.startswith(('temporal_actor.', 'temporal_critic.'))
                           for name, _ in policy.named_parameters())
        if len(saved_ids) == legacy_count and current_count > legacy_count:
            first_new_id = max(saved_ids, default=-1) + 1
            group['params'] = [*saved_ids, *range(first_new_id, first_new_id + current_count - legacy_count)]
        elif len(saved_ids) != current_count:
            raise ValueError('Checkpoint optimizer parameter layout is incompatible with the policy.')
    optimizer.load_state_dict(saved_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'play', 'evaluate'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--policy-architecture', choices=['legacy', 'mlp'], help='Fresh runs default to mlp; resumes keep the saved architecture')
    parser.add_argument('--teacher', action='store_true',
                        help='Encode frames with EmbeddingGemma-2 itself instead of the distilled student (much slower)')
    parser.add_argument('--student-encoder', type=Path, default=ROOT/'models/distillation/student.pt',
                        help='Distilled image encoder; runs on CPU')
    parser.add_argument('--calibration-frames', type=Path, default=ROOT/'models/distillation/frames.pt',
                        help='Distillation frames used to calibrate a fresh student policy')
    parser.add_argument('--headless', action='store_true', help='play without a window')
    parser.add_argument('--show', action='store_true', help='train or evaluate with the game window')
    parser.add_argument('--mode', choices=['sync', 'realtime'], default='sync',
                        help='sync pauses the emulator while encoding; realtime keeps it at --fps')
    parser.add_argument('--fps', type=float,
                        help='Emulator pacing; 0 runs sync mode as fast as possible '
                             '(default: 0 for train/evaluate, 60 for play)')
    parser.add_argument('--rollout', type=int, default=128)
    parser.add_argument('--total-frames', type=int, help='Explicit bounded diagnostic only; otherwise unbounded')
    parser.add_argument('--resume', action='store_true', help='train: continue --checkpoint instead of starting fresh')
    parser.add_argument('--checkpoint', type=Path,
                        help='default: checkpoints/latest.pt; play/evaluate use checkpoints/best.pt when it exists')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10,
                        help='Death-ended evaluation attempts before stopping')
    args = parser.parse_args()
    if args.checkpoint is None:
        best = ROOT/'checkpoints/best.pt'
        args.checkpoint = best if args.command in ('play', 'evaluate') and best.exists() else ROOT/'checkpoints/latest.pt'
    if args.fps is None:
        args.fps = 60.0 if args.command == 'play' or args.mode == 'realtime' else 0.0
    args.headless = (args.headless if args.command == 'play' else not args.show)
    if args.teacher or args.command == 'prepare':
        args.student_encoder = None  # prepare always encodes action texts with the teacher
    if args.fps < 0 or args.rollout < 2 or (args.total_frames is not None and args.total_frames <= 0):
        parser.error('fps must be non-negative, rollout >= 2, total-frames positive')
    if args.fps == 0 and args.mode != 'sync':
        parser.error('--fps 0 (unthrottled) requires --mode sync; realtime inference would fall behind')
    if args.command == 'evaluate' and (args.episodes <= 0 or args.total_frames is not None):
        parser.error('evaluate requires positive --episodes and stops only at death, not by frame count')
    best_path = args.checkpoint.with_name('best.pt' if args.checkpoint.stem == 'latest'
                                          else f'{args.checkpoint.stem}-best.pt')
    if args.command == 'train' and not args.resume:
        existing = [path for path in (args.checkpoint, best_path) if path.exists()]
        if existing:
            parser.error(f'{" and ".join(map(str, existing))} already exist(s). Continue with --resume, '
                         'or move them away to start fresh.')
    if (args.command in ('play', 'evaluate') or args.resume) and not args.checkpoint.exists():
        parser.error(f'{args.checkpoint} does not exist yet. Start training with `mario-agent train`.')
    directory = ROOT/'models/embeddinggemma-2'
    if args.command == 'prepare':
        download_model(directory)
    if args.student_encoder:
        from .student import StudentEncoder
        encoder = StudentEncoder(args.student_encoder, device=args.device)
    else:
        encoder = FrozenEncoder(directory, device=args.device)
    embeddings = prepare_actions(encoder, ROOT/'models/action-embeddings.pt')
    if args.command == 'prepare':
        print(f'Prepared {len(ACTIONS)} actions: {tuple(embeddings.shape)}')
        return
    payload = checkpoint.load(args.checkpoint) if args.resume or args.command in ('play', 'evaluate') else None
    architecture = args.policy_architecture or (payload['config'].get('policy_architecture', 'legacy') if payload else 'mlp')
    config = {'reward_scale': 15.0, 'image_tokens': 70, 'seed': args.seed,
              'environment': 'SuperMarioBros-full-pixel-v1', 'learning_rate': 3e-4,
              'gap_bonus': 1.0, 'gap_detector': 'overworld-rgb-v1',
              'death_penalty': 15.0, 'clear_reward': 50.0, 'pixels_per_reward': 32.0,
              'progress_clip': 5.0, 'reward_version': 'visual-frontier-v5', 'stall_timeout_frames': 1800,
              'restart_probability': 0.5, 'restart_backoff_px': 192.0, 'restart_snapshot_px': 64.0,
              'entropy_target': 0.5, 'entropy_coef_max': 0.1, 'entropy_adapt_rate': 0.5,
              'best_window_attempts': 40,
              'ppo_epochs': 4, 'ppo_minibatch_size': 32, 'target_kl': 0.02, 'value_clip': None,
              'discount_reference_frames': 20, 'optimizer_weight_decay': 0.0,
              'left_action_logit_bias': -3.0, 'entropy_coef': 0.01,
              'temporal_context': 'previous-embedding-delta-v1',
              'stagnation_threshold': 4, 'stagnation_deadband_px': 1.0,
              'recovery_actions': ['down', 'run_left', 'run_right_jump_long', 'right_jump_short'],
              'recovery_logit_bias': 8.0, 'recovery_down_hold_frames': 24,
              'recovery_progress_px': 8.0, 'recovery_progress_window': 4,
              'recovery_progress_events_required': 2, 'recovery_window_net_progress_px': 8.0, 'pipe_recovery': 'rgb-side-entry-v1'}
    config['policy_architecture'] = architecture
    config['feature_standardization'] = 'running-rollout-v1' if architecture == 'mlp' else 'identity'
    config['encoder_backend'] = 'student' if args.student_encoder else 'embeddinggemma2'
    config['student_fingerprint'] = encoder.fingerprint if args.student_encoder else None
    reward_objective_migrated = False
    policy_behavior_migrated = False
    recovery_behavior_migrated = False
    temporal_context_migrated = False
    if payload:
        # Preserve learned weights and optimizer while replacing old reward settings.
        saved_config = dict(payload['config'])
        saved_config.setdefault('policy_architecture', 'legacy')
        saved_config.setdefault('feature_standardization', 'identity')
        saved_config.setdefault('encoder_backend', 'embeddinggemma2')
        saved_config.setdefault('student_fingerprint', None)
        old_reward_version = saved_config.get('reward_version')
        old_left_bias = saved_config.get('left_action_logit_bias')
        old_entropy_coef = saved_config.get('entropy_coef')
        old_temporal_context = saved_config.get('temporal_context')
        recovery_keys = ('stagnation_threshold', 'stagnation_deadband_px',
                         'recovery_actions', 'recovery_logit_bias', 'recovery_down_hold_frames',
                         'recovery_progress_px', 'recovery_progress_window',
                         'recovery_progress_events_required', 'recovery_window_net_progress_px')
        recovery_behavior_migrated = any(saved_config.get(key) != config[key] for key in recovery_keys)
        for key in ('movement_reward', 'motion_detector'):
            saved_config.pop(key, None)
        saved_config.update({key: config[key] for key in (
            'reward_scale', 'death_penalty', 'clear_reward', 'pixels_per_reward', 'progress_clip', 'reward_version',
            'ppo_epochs', 'ppo_minibatch_size', 'target_kl', 'value_clip', 'discount_reference_frames',
            'optimizer_weight_decay', 'feature_standardization',
            'left_action_logit_bias', 'entropy_coef', 'temporal_context', 'stagnation_threshold',
            'stagnation_deadband_px', 'recovery_actions', 'recovery_logit_bias',
            'recovery_down_hold_frames', 'recovery_progress_px', 'recovery_progress_window',
            'recovery_progress_events_required', 'recovery_window_net_progress_px', 'stall_timeout_frames',
            'restart_probability', 'restart_backoff_px', 'restart_snapshot_px',
            'entropy_target', 'entropy_coef_max', 'entropy_adapt_rate', 'best_window_attempts')})
        recovery_behavior_migrated |= saved_config.get('pipe_recovery') != config['pipe_recovery']
        saved_config['pipe_recovery'] = config['pipe_recovery']
        saved_config.setdefault('gap_bonus', 1.0)
        saved_config.setdefault('gap_detector', 'overworld-rgb-v1')
        reward_objective_migrated = old_reward_version != config['reward_version']
        policy_behavior_migrated = (old_left_bias != config['left_action_logit_bias'] or
                                    old_entropy_coef != config['entropy_coef'])
        temporal_context_migrated = old_temporal_context != config['temporal_context']
        if {k: v for k, v in saved_config.items() if k != 'seed'} != {k: v for k, v in config.items() if k != 'seed'}:
            if saved_config['encoder_backend'] != config['encoder_backend']:
                raise ValueError(f'{args.checkpoint} was trained with the '
                                 f'{"teacher (pass --teacher)" if saved_config["encoder_backend"] == "embeddinggemma2" else "student encoder (omit --teacher)"}.')
            if saved_config['student_fingerprint'] != config['student_fingerprint']:
                raise ValueError(f'{args.checkpoint} was trained with a different student encoder than '
                                 f'{args.student_encoder}; pass the matching --student-encoder.')
            raise ValueError('Checkpoint training configuration is incompatible')
        config = saved_config
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    policy_kwargs = {}
    if architecture == 'mlp':
        policy_kwargs['architecture'] = 'mlp'
        if args.student_encoder:
            if payload:
                mean, std = payload['policy']['feature_mean'], payload['policy']['feature_std']
            else:
                from .student import calibrate_policy_features
                print('Calibrating policy inputs on training-stream student outputs...', flush=True)
                mean, std = calibrate_policy_features(encoder, args.calibration_frames)
            policy_kwargs.update(feature_mean=mean, feature_std=std)
    policy = ActorCritic(embeddings, left_action_logit_bias=config['left_action_logit_bias'], **policy_kwargs)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['optimizer_weight_decay'])
    state = {'frames': 0, 'decisions': 0, 'updates': 0, 'deaths': 0, 'clears': 0,
             'episode_seed': config['seed'], 'episode_commands': [], 'gaps': 0,
             'stalls': 0, 'stall_frames': 0, 'stall_frontier_px': 0.0, 'restarts': 0,
             'entropy_coef': config['entropy_coef'], 'episode_from_start': True,
             'recent_full_start_clears': [], 'best_clear_rate': -1.0,
             'stagnant_decisions': 0, 'recovery_active': False, 'recovery_action_index': 0,
             'recovery_progress_window': [], 'previous_features': None}
    if args.command == 'evaluate':
        state['episode_seed'] = args.seed
    if payload:
        temporal_weights_migrated = load_policy_weights(policy, payload['policy'])
        temporal_context_migrated = temporal_context_migrated or temporal_weights_migrated
        if args.command == 'evaluate':
            # Evaluate fresh, death-ended attempts under an explicit seed sequence.
            state['episode_commands'] = []
        else:
            load_optimizer_state(optimizer, payload['optimizer'], policy)
            if args.command == 'train' and reward_objective_migrated:
                reset_critic_value_scale(policy, optimizer)
                optimizer.state.clear()  # Old-objective momentum is also stale for the actor.
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate']
                group['weight_decay'] = config['optimizer_weight_decay']
            legacy_recovery_state = 'recovery_active' not in payload['training_state']
            state.update({k: v for k, v in payload['training_state'].items() if k != 'torch_rng_state'})
            if legacy_recovery_state and state['stagnant_decisions'] >= config['stagnation_threshold']:
                state['recovery_active'] = True
                state['recovery_action_index'] = (
                    (state['stagnant_decisions']-config['stagnation_threshold']) %
                    len(config['recovery_actions']))
            torch.set_rng_state(payload['training_state']['torch_rng_state'])
            if args.command == 'play':
                # Watch from the level start rather than continuing whatever (possibly
                # restarted) episode training was in when it saved; keep the counters.
                state.update(episode_commands=[], previous_features=None, visual_position_px=0.0,
                             visual_frontier_px=0.0, stall_frames=0, stall_frontier_px=0.0,
                             stagnant_decisions=0, recovery_active=False, recovery_action_index=0,
                             recovery_progress_window=[], episode_from_start=True)
    env = make_env()
    frame, _ = env.reset(seed=state['episode_seed'])
    detector = StageDetector()
    gap_detector = GapDetector()
    motion = RightwardProgress()
    legacy_frontier = bool(payload and args.command != 'evaluate' and
                           'visual_frontier_px' not in payload['training_state'])
    frontier = ProgressFrontier(state.get('visual_position_px', 0.0), state.get('visual_frontier_px', 0.0))
    motion.reset(frame)
    detector.observe(frame)
    gap_detector.observe(frame)
    # Deterministic replay reconstructs the current episode without reading RAM.
    for index in state['episode_commands']:
        frame, died = step(env, COMMANDS[index])
        if detector.observe(frame):
            gap_detector = GapDetector()
            motion.reset(frame)
            if legacy_frontier:
                frontier.reset()
        else:
            gap_detector.observe(frame)
        replay_progress = motion.observe(frame)
        if legacy_frontier:
            frontier.observe(replay_progress)
            frontier.claim()
        if died:
            raise ValueError('Saved episode replay unexpectedly terminated')
    # A resumed run starts a new decision; do not credit a discarded partial action.
    frontier.claim()
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
        return {**{k: state[k] for k in ('frames', 'decisions', 'updates', 'deaths', 'clears', 'gaps',
                                      'stalls', 'stall_frames', 'restarts',
                                      'stagnant_decisions', 'recovery_active', 'recovery_action_index',
                                      'recovery_progress_window')},
                'visual_position_px': frontier.position, 'visual_frontier_px': frontier.frontier}

    def save_checkpoint():
        state.update(visual_position_px=frontier.position, visual_frontier_px=frontier.frontier)
        # latest.pt is always the newest policy, which can be worse than an earlier one;
        # also keep the policy with the best level-clear rate over recent full attempts.
        # Score windows by a Wilson lower bound (z=1) so a lucky 40-attempt streak does
        # not lock in, require at least one clear, and let ties favour the newer policy.
        recent = state['recent_full_start_clears']
        new_best = None
        if len(recent) >= config['best_window_attempts'] and sum(recent) > 0:
            n, rate = len(recent), sum(recent)/len(recent)
            score = (rate + 1/(2*n) - math.sqrt(rate*(1-rate)/n + 1/(4*n*n))) / (1 + 1/n)
            if score >= state['best_clear_rate']:
                state['best_clear_rate'] = new_best = score
        checkpoint.save(args.checkpoint, policy, optimizer, config, state)
        if new_best is not None:
            checkpoint.save(best_path, policy, optimizer, config, state)
            emit('best_checkpoint', clear_rate=sum(recent)/len(recent), lower_bound=new_best, path=str(best_path))

    def record_full_attempt(cleared_level):
        # Only attempts that began at the level start measure whole-level skill;
        # restarted attempts begin next to the obstacle.
        if state['episode_from_start']:
            state['recent_full_start_clears'].append(int(cleared_level))
            del state['recent_full_start_clears'][:-config['best_window_attempts']]
        state['episode_from_start'] = False

    # Restart curriculum (training only): emulator snapshots taken every few pixels
    # of new ground let a failed episode resume shortly before the place it died, so
    # hard obstacles get many attempts without replaying the level from the start.
    # Snapshots carry reward detectors and the controller-command prefix, so rewards
    # are not paid twice and checkpoint replay still reconstructs the episode.
    restarts_enabled = args.command == 'train' and config['restart_probability'] > 0
    episode_snapshots = []
    next_snapshot_px = config['restart_snapshot_px']

    def capture_snapshot():
        episode_snapshots.append(dict(
            emulator=env.unwrapped.dump_state(), frame=frame.copy(),
            commands=list(state['episode_commands']), seed=state['episode_seed'],
            position=frontier.position, frontier=frontier.frontier,
            motion=deepcopy(motion), detector=deepcopy(detector), gap_detector=deepcopy(gap_detector)))

    def reset_recovery():
        state['stagnant_decisions'] = 0
        state['recovery_active'] = False
        state['recovery_action_index'] = 0
        state['recovery_progress_window'] = []

    def finish(raw, done, bootstrap=0.0, outcome='step'):
        nonlocal active, last_reward, metrics, save_due
        new_progress = 0.0
        if active is not None and outcome == 'step':
            new_progress = frontier.claim()
            raw = reward(False, False, new_progress, active['gap_bonus'],
                         pixels_per_reward=config['pixels_per_reward'], progress_clip=config['progress_clip'],
                         deadband_px=0.0)
        last_reward = raw
        if outcome in ('death', 'stall', 'level_clear'):
            reset_recovery()
        if args.command == 'evaluate':
            # Count terminal rewards even if the death/transition lands while
            # the controller has no active policy decision to credit.
            evaluation['episode_raw_reward'] += raw
            if active is not None:
                evaluation['episode_progress_px'] += active['right_progress']
        if active is not None:
            recovery_evidence_events = 0
            recovery_window_net_progress = 0.0
            if outcome not in ('death', 'stall', 'level_clear'):
                if state['recovery_active']:
                    window = state['recovery_progress_window']
                    window.append(active['right_progress'])
                    del window[:-config['recovery_progress_window']]
                    positive_events = sum(value >= config['recovery_progress_px'] for value in window)
                    net_progress = sum(window)
                    recovery_evidence_events = positive_events
                    recovery_window_net_progress = net_progress
                    escaped = (active['gap_bonus'] > 0 or
                               (len(window) >= config['recovery_progress_window'] and
                                positive_events >= config['recovery_progress_events_required'] and
                                net_progress >= config['recovery_window_net_progress_px']))
                    if escaped:
                        reset_recovery()
                    else:
                        state['recovery_action_index'] = (
                            (state['recovery_action_index']+1) % len(config['recovery_actions']))
                elif (active['right_progress'] <= config['stagnation_deadband_px'] and
                      active['gap_bonus'] <= 0):
                    state['stagnant_decisions'] += 1
                    if state['stagnant_decisions'] >= config['stagnation_threshold']:
                        state['recovery_active'] = True
                        state['recovery_action_index'] = 0
                        state['recovery_progress_window'] = []
                else:
                    reset_recovery()
            emit('decision_result', decision=state['decisions'], action=ACTIONS[active['action']].name,
                 duration_frames=max(1, state['frames']-active['started']),
                 input_age_frames=state['frames']-active['observed'],
                 displacement_px=round(active['right_progress'], 3), gap_bonus=active['gap_bonus'],
                 new_frontier_px=round(new_progress, 3),
                 executed_frames=max(1, state['frames']-active['started']),
                 recovery_evidence_events=recovery_evidence_events,
                 recovery_window_net_progress_px=round(recovery_window_net_progress, 3),
                 raw_reward=raw, scaled_reward=raw/config['reward_scale'], outcome=outcome,
                 recovery_action=active['recovery_action'], stagnant_decisions=state['stagnant_decisions'])
            if args.command == 'train':
                rollout.append(Transition(active['features'], active['action'], active['log_prob'],
                                          active['value'], raw/config['reward_scale'], done,
                                          active['action_bias'], active['previous_features'],
                                          max(1, state['frames']-active['started'])))
        active = None
        if len(rollout) >= args.rollout:
            update_started = time.perf_counter()
            rollout_features = torch.stack([item.features for item in rollout])
            seeded_statistics = payload is None and float(getattr(policy, 'feature_count', 1)) == 0
            if seeded_statistics:
                # Fresh run: zero-initialized output layers make logits/values independent
                # of statistics, so seeding them now keeps stored log-probs valid and the
                # first update already trains on standardized inputs.
                policy.update_feature_statistics(rollout_features)
            metrics = update(policy, optimizer, rollout, bootstrap,
                             entropy_coef=state['entropy_coef'], epochs=config['ppo_epochs'],
                             minibatch_size=config['ppo_minibatch_size'], target_kl=config['target_kl'],
                             reference_frames=config['discount_reference_frames'],
                             value_clip=config['value_clip'])
            # Statistics change only between updates, after old log-probs are consumed.
            if not seeded_statistics:
                policy.update_feature_statistics(rollout_features)
            metrics['update_ms'] = (time.perf_counter()-update_started)*1000
            metrics['feature_count'] = int(getattr(policy, 'feature_count', 0))
            # Entropy floor: raise the bonus while the policy is more certain than the
            # target, relax it back toward the base coefficient otherwise. A fixed 0.01 let
            # entropy fall to ~0.1 nats, after which the policy stopped trying alternatives.
            metrics['entropy_coef'] = state['entropy_coef']
            if metrics.get('actor_entropy') is not None:
                adjusted = state['entropy_coef'] * math.exp(
                    config['entropy_adapt_rate'] * (config['entropy_target'] - metrics['actor_entropy']))
                state['entropy_coef'] = min(max(adjusted, config['entropy_coef']), config['entropy_coef_max'])
            rollout.clear()
            state['updates'] += 1
            save_due = True
            emit('ppo', **metrics)

    def encode(image):
        before = time.perf_counter()
        features = encoder.encode_frame(image)
        return features, (time.perf_counter()-before)*1000

    emit('started', mode=args.mode, policy_architecture=architecture, feature_standardization=config['feature_standardization'], gap_bonus=config['gap_bonus'], reward_objective_migrated=reward_objective_migrated,
         reward_version=config['reward_version'], clear_reward=config['clear_reward'],
         pixels_per_reward=config['pixels_per_reward'], ppo_epochs=config['ppo_epochs'],
         target_kl=config['target_kl'], discount_reference_frames=config['discount_reference_frames'],
         critic_reset_for_reward_migration=bool(args.command == 'train' and payload and reward_objective_migrated),
         policy_behavior_migrated=policy_behavior_migrated,
         recovery_behavior_migrated=recovery_behavior_migrated,
         temporal_context=config['temporal_context'], temporal_context_migrated=temporal_context_migrated,
         left_action_logit_bias=config['left_action_logit_bias'], entropy_coef=state['entropy_coef'],
         stagnation_threshold=config['stagnation_threshold'], recovery_actions=config['recovery_actions'],
         recovery_logit_bias=config['recovery_logit_bias'],
         recovery_down_hold_frames=config['recovery_down_hold_frames'],
         recovery_progress_px=config['recovery_progress_px'],
         recovery_progress_window=config['recovery_progress_window'],
         recovery_progress_events_required=config['recovery_progress_events_required'],
         recovery_window_net_progress_px=config['recovery_window_net_progress_px'])
    print(f'{args.command}: {len(ACTIONS)} actions; mode={args.mode}; temporal=embedding delta; '
          f'PPO rollout={args.rollout}; encoder={encoder.device}; policy={architecture}', flush=True)
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
                    bootstrap = float(policy(features, previous_features=state['previous_features'])[1])
                if active is not None:
                    finish(0.0, False, bootstrap)
                else:
                    # Startup/transition animation while no action exists must
                    # not become reward for the first subsequently chosen action.
                    frontier.claim()
                recovery_action = None
                action_bias = None
                pipe_plan = PipeExitRecovery().choose(observation[2]) if state['recovery_active'] else None
                if state['recovery_active']:
                    index = state['recovery_action_index']
                    recovery_action = pipe_plan[0] if pipe_plan else config['recovery_actions'][index]
                    action_bias = torch.zeros(len(ACTIONS))
                    recovery_index = next(i for i, item in enumerate(ACTIONS) if item.name == recovery_action)
                    action_bias[recovery_index] = config['recovery_logit_bias']
                previous_features = state['previous_features']
                action, log_prob, value, probabilities = policy.act(
                    features, sample=args.command == 'train', action_bias=action_bias,
                    previous_features=previous_features)
                commands = ACTIONS[action].commands()
                if pipe_plan and ACTIONS[action].name == pipe_plan[0]:
                    commands = ACTIONS[action].commands()[:1]*pipe_plan[1]
                    recovery_action = pipe_plan[2]
                if recovery_action == 'down' and ACTIONS[action].name == 'down':
                    commands = (('down',),)*config['recovery_down_hold_frames']
                active = dict(features=features, previous_features=previous_features,
                              action=action, log_prob=log_prob, value=value,
                              commands=commands, position=0, observed=observation[1], started=state['frames'], gap_bonus=0.0,
                              right_progress=0.0, recovery_action=recovery_action, action_bias=action_bias)
                state['previous_features'] = features.detach().clone()
                policy_frame = observation[2]
                state['decisions'] += 1
                emit('decision', action=ACTIONS[action].name, inference_ms=inference_ms,
                     observation_age_frames=state['frames']-observation[1], value=value,
                     recovery_action=recovery_action, stagnant_decisions=state['stagnant_decisions'])
            command = ()
            if active is not None:
                position = active['position']
                # Respect the macro duration while the next embedding is pending.
                # Holding the final direction indefinitely can walk into a gap.
                command = active['commands'][position] if position < len(active['commands']) else ()
                active['position'] += 1
            frame, died = step(env, command)
            state['frames'] += 1
            state['episode_commands'].append(COMMANDS.index(command))
            right_progress = motion.observe(frame)
            frontier.observe(right_progress)
            if right_progress is not None and active is not None:
                active['right_progress'] += right_progress
            cleared = detector.observe(frame)
            crossed = gap_detector.observe(frame)
            # A training episode that reaches no new visual ground for the timeout
            # ends like a death: the game timer would kill Mario anyway, but thousands
            # of frames later, where discounting hides the penalty and samples are wasted.
            if frontier.frontier > state['stall_frontier_px']:
                state['stall_frontier_px'] = frontier.frontier
                state['stall_frames'] = 0
            else:
                state['stall_frames'] += 1
            stalled = (args.command == 'train' and not died and not cleared and
                       state['stall_frames'] >= config['stall_timeout_frames'])
            if restarts_enabled and not (died or cleared or stalled) and frontier.frontier >= next_snapshot_px:
                capture_snapshot()
                next_snapshot_px = frontier.frontier + config['restart_snapshot_px']
            if crossed and not died and not cleared and active is not None:
                active['gap_bonus'] += config['gap_bonus']
                state['gaps'] += 1
                image_path = logdir/f'gap-{state["frames"]}.png'
                Image.fromarray(frame).save(image_path)
                emit('gap_crossing', bonus=config['gap_bonus'], captured_frame=str(image_path))
            if died or cleared or stalled:
                outcome = 'death' if died else 'stall' if stalled else 'clear'
                ended_frontier = frontier.frontier
                image_path = logdir/f'{outcome}-{state["frames"]}.png'
                Image.fromarray(frame).save(image_path)
                finish(reward(died or stalled, cleared, death_penalty=config['death_penalty'],
                              clear_reward=config['clear_reward']), True,
                       outcome='level_clear' if outcome == 'clear' else outcome)
                frontier.reset()
                state['stall_frames'] = 0
                state['stall_frontier_px'] = 0.0
                state['previous_features'] = None
                state[{'death': 'deaths', 'stall': 'stalls', 'clear': 'clears'}[outcome]] += 1
                record_full_attempt(cleared_level=outcome == 'clear')
                emit(outcome, raw_reward=last_reward, captured_frame=str(image_path))
                generation += 1
                gap_detector = GapDetector()
                if cleared:
                    motion.reset(frame)
                    # Snapshots and their frontier coordinates belong to the finished level.
                    episode_snapshots.clear()
                    next_snapshot_px = config['restart_snapshot_px']
                if died or stalled:
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
                        candidates = [snap for snap in episode_snapshots
                                      if snap['frontier'] <= ended_frontier - config['restart_backoff_px']]
                        if (restarts_enabled and candidates and
                                float(torch.rand(())) < config['restart_probability']):
                            snapshot = candidates[-1]
                            del episode_snapshots[episode_snapshots.index(snapshot)+1:]
                            env.unwrapped.load_state(snapshot['emulator'])
                            frame = snapshot['frame'].copy()
                            state['episode_seed'] = snapshot['seed']
                            state['episode_commands'] = list(snapshot['commands'])
                            frontier.position = snapshot['position']
                            frontier.frontier = state['stall_frontier_px'] = snapshot['frontier']
                            motion = deepcopy(snapshot['motion'])
                            detector = deepcopy(snapshot['detector'])
                            gap_detector = deepcopy(snapshot['gap_detector'])
                            next_snapshot_px = snapshot['frontier'] + config['restart_snapshot_px']
                            state['restarts'] += 1
                            state['episode_from_start'] = False
                            emit('restart', from_frontier_px=snapshot['frontier'], ended_frontier_px=ended_frontier)
                        else:
                            state['episode_seed'] += 1
                            state['episode_commands'] = []
                            frame, _ = env.reset(seed=state['episode_seed'])
                            motion.reset(frame)
                            detector = StageDetector()
                            detector.observe(frame)
                            episode_snapshots.clear()
                            next_snapshot_px = config['restart_snapshot_px']
                            state['episode_from_start'] = True
            if stop_evaluation:
                break
            if args.command == 'train' and save_due:
                save_checkpoint()
                save_due = False
            elapsed = max(time.perf_counter()-started, .001)
            fps = (state['frames']-start_frames)/elapsed
            dps = (state['decisions']-start_decisions)/elapsed
            if display:
                recovery = active['recovery_action'] if active and active['recovery_action'] else 'off'
                recovery = {'run_right_jump_long': 'R+JUMP-LONG',
                            'right_jump_short': 'R+JUMP-SHORT',
                            'run_left': 'RUN-LEFT'}.get(recovery, recovery.upper())
                recovery_evidence = sum(value >= config['recovery_progress_px']
                                        for value in state['recovery_progress_window'])
                recovery_net = sum(state['recovery_progress_window'])
                lines = [f'{args.command} / {args.mode} / {architecture}', f'FPS {fps:.1f}  decisions/s {dps:.2f}',
                         f'Frame {state["frames"]}', f'Action: {ACTIONS[active["action"]].name if active else "waiting"}',
                         f'{"Student" if args.student_encoder else "Gemma"} {inference_ms:.1f} ms', f'Input age {state["frames"]-active["observed"] if active else 0} frames',
                         f'Stagnant decisions {state["stagnant_decisions"]}',
                         f'Recovery {recovery}  {recovery_evidence}/{config["recovery_progress_events_required"]}  {recovery_net:+.0f}px',
                         f'New ground {frontier.frontier:.0f}px',
                         f'Rollout {len(rollout)}/{args.rollout}, actor {metrics.get("actor_samples", 0)}',
                         f'Updates {state["updates"]} KL {metrics.get("final_kl", 0):.3g}',
                         f'Dead {state["deaths"]} stalled {state["stalls"]} clear {state["clears"]} gaps {state["gaps"]} restarts {state["restarts"]}', f'Reward {last_reward:g} / {last_reward/config["reward_scale"]:g}',
                         *[f'{k}: {metrics[k]:.4g}' for k in ('policy_loss', 'value_loss', 'entropy', 'explained_variance') if isinstance(metrics.get(k), (float, int))],
                         'Recovery probabilities:' if active and active['recovery_action'] else 'Policy probabilities:',
                         *[f'{ACTIONS[i].name}: {probabilities[i]:.2f}' for i in sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]],
                         'Below: policy input frame']
                if not display.draw(frame, policy_frame, lines):
                    break
            if time.perf_counter()-last_report >= 2:
                print(f'frames={state["frames"]} fps={fps:.1f} decisions/s={dps:.2f} updates={state["updates"]} deaths={state["deaths"]} reward={last_reward:g}', flush=True)
                last_report = time.perf_counter()
            if args.fps > 0:
                next_tick = max(next_tick + 1/args.fps, time.perf_counter())
                time.sleep(max(0, next_tick-time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        worker.shutdown(wait=True, cancel_futures=True)
        if args.command == 'train':
            frontier.claim()
            save_checkpoint()
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
