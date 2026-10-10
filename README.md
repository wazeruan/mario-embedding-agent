# Mario EmbeddingGemma-2 PPO agent

Fresh rewrite in `/Users/weiciruan/Documents/mario-embedding-agent`.

A frozen image/text **google/embeddinggemma-2** encoder or its distilled CNN maps each RGB game screenshot to a normalized 768-dimensional vector. Action descriptions are encoded once and cached. The base actor starts as a linear projection initialized from those action vectors, with a linear temporal residual. Fresh runs additionally use a nonlinear residual actor on the joint current embedding and its change from the previous decision. The critic has separate nonlinear current-frame and temporal heads. PPO trains these policy/value heads while the encoder and action prototypes stay frozen. A categorical softmax samples actions during training; play and evaluation use argmax. No game RAM or numerical game state enters the policy.

Temporal context spans policy decisions rather than every emulator frame: it can reveal visual motion or a scene change that a single screenshot cannot. The exact prior embedding is stored with each PPO transition so training recomputes the same action probability, and it is checkpointed for continuation. Death and detected WORLD transitions clear this context. Existing direct-actor checkpoints migrate with zero-initialized temporal heads, so their starting behavior is preserved while PPO learns whether the new signal helps.

The pinned [official encoder](https://huggingface.co/google/embeddinggemma-2) revision is `914f7f89142e33e77833254d9c9b90c3cef7303b`; image budget 70 soft tokens, audio disabled. Model files and action vectors are local under `models/` and ignored by Git.

## Run

```sh
cd /Users/weiciruan/Documents/mario-embedding-agent
uv sync
uv run mario-agent prepare          # download EmbeddingGemma-2 and encode the action texts
uv run mario-agent train            # start a new run in checkpoints/latest.pt
uv run mario-agent train --resume   # continue it
uv run mario-agent play             # watch the latest checkpoint
uv run mario-agent evaluate         # deterministic, headless check of the latest checkpoint
```

Defaults: frames are encoded by the distilled IMPALA student (`models/distillation/student.pt`, CPU, ~3 ms) and the policy decides on the current screenshot (`--mode sync`). Training and evaluation run headless and unthrottled (about 650 FPS on an M4); play opens the game window at 60 FPS. Training saves `checkpoints/latest.pt` after each PPO update and `checkpoints/best.pt` whenever its level-clear rate reaches a new high; `play` and `evaluate` use `best.pt` when it exists, otherwise `latest.pt`, so they can watch a run while it trains. `train` refuses to overwrite existing checkpoints: continue with `--resume`, or move both files away to start fresh. Stop training with a single Ctrl+C; it saves before exiting.

Options: `--teacher` encodes with EmbeddingGemma-2 itself (~160 ms per frame on MPS); `--checkpoint PATH` selects another run; `--show` displays the window while training; `--fps 30` slows play; `--mode realtime` keeps the emulator running during inference, so the policy sees screenshots several frames old. `--total-frames N` bounds a diagnostic run. A fresh student run calibrates its input statistics on `--calibration-frames` (default `models/distillation/frames.pt`). Models must be prepared before train/play/evaluate.

Each action is a button macro: short jumps hold A for 4 emulator frames and release it for 2; long jumps hold A for 20 and release for 2. While waiting for the next decision, all buttons are released after the macro, so inference cannot accidentally extend a jump.

## Actions and objective

There are 18 actions: idle, walk/run right, walk/run left, down, up, fire, and short/long jumps vertically, right, running right, left, running left. They are button-duration macros, not guaranteed successful maneuvers; jump height also depends on game physics and current movement. The policy applies a −3 logit prior to actions containing left, reducing their baseline softmax mass; frame-conditioned logits can still select left when their evidence overcomes that prior. Training starts with an entropy coefficient of 0.01 and adapts it (see Entropy floor).

To break a visual stall, the controller counts consecutive completed actions whose RGB-estimated rightward displacement is at most one pixel and which did not cross a detected gap. After four, it biases the policy through a recovery sequence: hold `down` for 24 frames to try entering a pipe, back away (`run_left`), then jump/run right. One brief movement burst no longer exits recovery: it takes two progress-bearing actions in a four-action window with positive net displacement, or a detected gap crossing, to confirm escape. Death or a rendered level transition resets recovery. This uses rendered-frame motion only. The recovery stage and progress window are saved in checkpoints. The temporary action bias is stored with that PPO transition and reapplied during PPO updates, so the sampled-action probability remains consistent with the probability used for training. The UI and JSONL logs report the stagnation count, recovery action, and accumulated escape evidence.

For an ordinary policy decision, movement reward pays for newly reached visual ground: `clip(new_frontier_pixels / 32, 0, 5)`. A stateful frontier integrates Mario sprite movement and scenery scroll from RGB frames. Advancing 32 pixels beyond the previous frontier earns +1; retreating or retracing already visited ground earns zero, so a repeated left/right cycle cannot farm reward. There is no per-decision penalty or enemy-elimination reward. A confirmed gap crossing adds +1. Death gives −15 and a detected WORLD transition gives +50; either terminal outcome overrides movement and gap rewards, and death takes precedence. The frontier resets on death or WORLD transition. This remains a visual heuristic: tracking errors can still shift the frontier, and missing sprites contribute no guessed motion.

PPO divides rewards by 15 (death is −1, 32 px of new ground ≈ +0.067) and collects 128-decision rollouts by default (`--rollout`). Each rollout receives up to four epochs with minibatches of 32, AdamW at 0.0003 with zero weight decay, policy clip 0.2, unclipped value loss, discount 0.99 and GAE lambda 0.95 per 20 emulator frames, and an adaptive entropy coefficient (0.01–0.1). Each transition stores its actual elapsed frames, including time waiting for the next embedding. Discounts and GAE traces use `gamma ** (frames / 20)` and `lambda ** (frames / 20)`, so short and long actions use the same game-time horizon. Actor loss and entropy use only unassisted decisions; recovery decisions still train the critic. A KL guard stops the update when estimated policy divergence exceeds 1.5 times the 0.02 target, following the [PPO early-stopping practice](https://spinningup.openai.com/en/latest/algorithms/ppo.html). This protects update size; it does not establish improved gameplay.

The controller releases buttons after a macro ends while encoding is pending, rather than extending the last directional button indefinitely. The GUI remains at 60 FPS. JSONL logs record actual action duration, new-frontier reward, recovery evidence, actor/recovery sample counts, optimizer steps, KL stopping and update latency. Checkpoints persist the visual position/frontier; old checkpoints reconstruct it by replaying recorded commands and analyzing RGB frames. Changing the reward objective preserves learned actor weights and critic feature layers, resets critic output scale and clears stale optimizer moments. Previously discarded partial actions do not earn reward on resume.

Only RGB screenshots reach the policy and success detector. The emulator adapter exposes the permitted death-only `terminated` flag and suppresses game rewards/info and RAM-based animation skipping. The policy never reads score, position, lives, time, or other game metadata. Death frames are saved before reset. Level success is detected by a stable change in the rendered WORLD number over three frames; the game continues into the next level without reset. This detects intermediate level transitions, not final-game victory. Enemy elimination receives no reward.

## Training, logs and continuation

The game window shows action, inference latency, observation age, rollout fill, updates, deaths, clears, raw/scaled reward, losses, entropy, explained variance and top action probabilities. The smaller image is the screenshot actually encoded for the current decision.

`runs/train.jsonl` records decisions, PPO updates and outcomes; `runs/death-*.png` and `runs/clear-*.png` retain outcome screenshots. `checkpoints/latest.pt` stores actor, critic, temporal heads, optimizer, RNG, counters, prior embedding and the current episode's controller commands. Resume deterministically replays that episode from its reset seed using only screenshots and controller commands. An unfinished rollout/action is discarded at stop and a fresh decision begins on resume. The seed saved in the checkpoint takes precedence over `--seed` on resume. FPS, mode and rollout can be changed when continuing. Replay time grows with episode length.

Cached RGB frames avoid repeated encoder work only when the frame is byte-identical. PPO reuses detached features and never reruns the encoder for gradient epochs. Model latency and cache-hit latency are separate measurements. Faster training and pretrained embeddings do not guarantee level completion; gameplay improvement requires evaluation across episodes.

### Visual progress and gap rewards

Rightward progress is estimated from Mario sprite movement plus horizontal scenery scroll between RGB frames. Button presses alone do not earn reward. If Mario is not detected in a frame, the movement baseline resets; later observations cannot carry missing-frame displacement into a different policy action. Ambiguous scenery matches contribute zero estimated scroll; detectable Mario screen movement is still counted. Gap crossings are recognized from a sequence of RGB frames: a grounded approach to a visible hole at the ground row, airborne passage, then three grounded frames beyond the far edge. A crossing is counted once while its gap remains tracked, and its +1 is added to the rightward movement reward for that policy decision. This conservative detector supports the original small-Mario, blue-sky overworld palette and ground height; other palettes, large/fire Mario, obscured sprites or gaps leaving the viewport may be missed. It is a heuristic, not general scene understanding. `runs/gap-*.png` records detected landings and `gap_crossing` log events report the +1 bonus. The UI shows the gap count.

Resuming a checkpoint from an earlier reward objective preserves actor weights, critic feature layers, episode counters; it migrates the reward scale and death penalty, applies the left-action prior and lower entropy coefficient, and starts fresh rollouts. Because value targets change with the reward objective, migration zeros both critic output layers and clears optimizer moments for all policy parameters while retaining learned critic features. The `started` event records reward and temporal-context migrations. Detector history is reconstructed during episode replay without awarding bonuses again.

## Distilled image encoder (experimental)

The student CNN learns the frozen teacher's 768-dimensional normalized image embeddings from real Mario screenshots. It predicts standardized embedding residuals using statistics computed only on training frames. This prevents dominant shared embedding directions from disguising a constant-output student. Action text vectors remain pre-encoded by the teacher; the student accepts images only. PPO continues to train its actor/critic from detached student features; it does not update the encoder online.

```sh
uv run mario-agent prepare
uv run mario-distill collect --samples 6144 --seed 1 --data models/distillation/frames-new.pt
uv run mario-distill train --data models/distillation/frames-new.pt --output models/distillation/student-new.pt
uv run mario-distill benchmark --data models/distillation/frames-new.pt --output models/distillation/student-new.pt
```

Collection interleaves four independently seeded emulator streams, retains original RGB frames and teacher vectors, and resets each environment only after death. Exact duplicate frames are removed globally. Stream 3 is held out entirely, including all its episodes; this is a small pilot dataset rather than comprehensive gameplay coverage. Both training and benchmark reports compare the student with the training-mean embedding baseline. The benchmark measures uncached, warmed inference on the same held-out screenshots and writes `models/distillation/student.benchmark.json`; embedding agreement is not evidence of improved gameplay.

The distilled student is the default encoder (`--teacher` uses EmbeddingGemma-2 directly). Student inference runs on CPU and never loads the teacher; keep the prepared action cache. A checkpoint resumes or plays only with the encoder it was trained with: the same backend and, for students, the identical student file fingerprint. Re-distilling creates a new encoder identity and needs a fresh PPO run, so `mario-distill collect` and `train` refuse to overwrite existing files.

### Local pilot results (2026-10-08)

Seed 0 collection yielded 1,024 globally unique real screenshots across 25 stream/episode groups: 769 training and 255 held-out stream-3 frames. After 60 epochs, the best student checkpoint had held-out cosine 0.99492 and standardized RMSE 0.86749, versus 0.99346 and 0.97214 for the constant training-mean baseline (about 11% lower error). Pairwise-distance correlation was 0.597, so frame geometry is only partially preserved.

On 32 evenly spaced held-out screenshots, warmed uncached inference including preprocessing and CPU feature return measured median 0.735 ms / p95 0.778 ms for the CPU student, versus 162.907 ms / 186.460 ms for the MPS teacher. This is approximately 222× faster for the encoder on this machine; it does not imply 222× faster decisions or PPO. The student has 499,728 parameters and its artifact is about 2 MB. Benchmark-subset cosine was 0.99473, standardized RMSE 0.88020 versus the mean baseline's 0.96241, and pairwise-distance correlation 0.501. Action-prototype top-1 agreement was 100% on this small subset; this is retrieval agreement, not PPO action agreement or game success. Gameplay improvement has not been evaluated.

### Larger distillation set and finer student (2026-10-09)

The 769-frame pilot student lost most information about whether Mario is airborne. Linear probes on held-out stream 3 of a new 6,144-frame collection (`--seed 1`) predict pixel-derived labels: Mario screen x (R²), airborne and obstacle-ahead (accuracy; majority baselines 0.63 and 0.62).

| Features | Mario x R² | Airborne | Obstacle | Std. RMSE vs teacher | Distance corr. |
|---|---|---|---|---|---|
| EmbeddingGemma-2 | 0.71 | 0.95 | 0.87 | — | — |
| Pilot student (769 frames, 4×4 pool) | 0.81 | 0.72 | 0.78 | 0.88 | 0.50 |
| 6k student, 4×4 pool | 0.85 | 0.96 | 0.89 | 0.57 | 0.78 |
| 6k student, 8×8 pool | 0.81 | 0.94 | 0.87 | 0.54 | 0.83 |

In 300k-frame PPO runs (seed 0, sync, unthrottled), median per-episode visual frontier over the last 50k frames was 698 px for the pilot student (its policy entropy had collapsed), 1,083 px for the 6k 4×4 student and 1,323 px for the 6k 8×8 student (259 heuristic gap detections). This is one seed per configuration. (These plain-CNN students were later replaced by the IMPALA student below.)


The 6k set and the IMPALA student are now the defaults (`models/distillation/frames.pt`, `student.pt`); the pilot data and the plain CNN students were retired.

### Student CNN architecture (2026-10-09)

Three students were distilled on the same 6,144-frame set and trained with PPO for 300k frames (sync, unthrottled, current defaults including restarts):

| Student | Params | CPU ms/frame | Std. RMSE vs teacher | Seed 0 clears | Seed 1 clears |
|---|---|---|---|---|---|
| small, 24/48/64 channels, 8×8 pool | 1.29M | 0.8 | 0.536 | 26 | 14 |
| small, 2× width | 2.45M | 1.2 | 0.533 | 26 | — |
| IMPALA CNN, 16/32/64, 8×8 pool | 1.46M | 2.7 | 0.532 | **42** | **22** |

All three imitate the teacher about equally and match it on the linear probes; doubling width gave nothing. The IMPALA encoder (conv, max-pool and two residual blocks per stage; Espeholt et al., 2018) gave more level clears on both seeds and a better-fitting critic (late explained variance 0.55–0.59 vs 0.13–0.35). The plain CNN was then removed: `mario-distill train` always builds the IMPALA student (`--width` scales its channels), and only IMPALA student files load.


### Stall timeout

Training ends an episode that reaches no new visual ground for 1,800 frames (30 s at 60 FPS) with outcome `stall`, the death penalty and a reset to the next seed. Without it, a policy stuck behind an obstacle waited about 8,000 frames for the game timer; that late penalty is nearly invisible after discounting and the frames are wasted. Evaluation and play never stall-terminate. The decision is based only on the RGB-derived frontier.

### Restart curriculum

Every attempt at a late obstacle used to cost a full run from the level start: a policy that reliably reached the staircase pit at ~2,390 px in World 1-1 got only 30–60 tries there per 100k frames. Training now captures an exact emulator snapshot (`dump_state`, ~0.04 ms) every 64 px of new visual ground. After a death or stall, with probability 0.5 (originally 0.75) it resumes from the latest snapshot at least 192 px before where the episode ended; otherwise it resets to the level start, so whole-level play keeps being practised. Snapshots carry the reward detectors and the controller-command prefix, so ground is not paid twice within an attempt and checkpoint replay reconstructs the restored episode. Level clears and full resets discard snapshots. Only RGB-derived progress chooses the restart point; evaluation and play never restart.

Continuing a checkpoint stuck at that pit for 200k more frames (seed 0) gave 43 level clears with restarts versus 26 without; the estimated success rate at the pit rose from 27% to 39%. One seed and a short run: treat this as a sample-efficiency improvement, not a guarantee.

### Entropy floor and best checkpoint

A 5.8M-frame run peaked near 1M frames and then degraded: policy entropy fell from 1.46 to 0.11 nats under the fixed 0.01 coefficient, stalls in front of the final staircase pit grew, later full attempts died early in the level, and `latest.pt` kept overwriting the better policy. Three changes address this:

- **Entropy floor.** After each update the entropy coefficient is multiplied by `exp(0.5 × (0.5 − actor entropy))` and clamped to [0.01, 0.1], so it rises while the policy is more certain than 0.5 nats and relaxes otherwise. The entropy used is measured on actor samples after the update.
- **Fewer near-failure restarts.** The restart probability drops from 0.75 to 0.5 so the opening of a level keeps being practised.
- **Best checkpoint.** Training tracks the last 40 attempts that began at a level start (a clear counts 1, a death or stall 0; restarted attempts are excluded). When the window contains a clear and its Wilson lower bound (z = 1) is at least the best so far, the policy is also saved as `best.pt`. This is a stochastic-policy proxy: `play` acts greedily and can differ.

Fresh 2M-frame runs (2026-10-09): without these changes (seed 0), the share of level-start attempts that cleared World 1-1 peaked at 39% around 1.25M frames and fell to 1% by 2M as entropy dropped to ~0.2; its final policy reached 725 px in greedy play. With them, seed 0 held 21–28% from 1M to 2M (entropy ~0.6) and seed 1 peaked at 37% and ended at 14% (entropy ~0.9). The `best.pt` of both seeds (saved at 1.05M and 1.29M frames) cleared World 1-1 in greedy `evaluate`; their final `latest.pt` did not. These runs used an earlier entropy reading that averaged in minibatches without actor samples; the committed adapter uses the post-update actor entropy.

### Underground exit-pipe recovery

The generic recovery cycle can leave Mario on top of the low pipe beside the underground room's green wall: a four-frame retreat followed by another right jump does not reliably clear the pipe. During stagnation, an RGB-only detector now recognizes the black room, tall right-side green wall, low adjoining pipe, and Mario sprite. It retreats left in 16-frame segments until Mario clears the edge, releases buttons in 8-frame segments while falling, then walks right in 24-frame segments into the pipe's left-facing side entrance. Each decision rechecks the screenshot; unsupported or ambiguous scenes retain generic recovery. No game metadata or additional rewards are used.

A replay reconstructed from recorded controller decisions reproduced the stuck room. Three left segments and two right segments entered the pipe; 180 neutral frames then returned Mario to the overworld without death. This demonstrates this recovery case, not general game completion. It is an explicit controller override: its samples train the critic but remain excluded from PPO actor updates. The UI labels overridden probabilities as recovery probabilities. Existing checkpoints resume with the new recovery version.


### Nonlinear policy and input calibration

Fresh training defaults to `--policy-architecture mlp`. The nonlinear actor uses `[standardized current embedding, standardized embedding delta] → LayerNorm → Linear(1536,256) → GELU → Linear(256,128) → GELU → Linear(128,18)`, added to the existing action-prototype projection. Its final layer starts at zero, preserving the pre-encoded action prior. The critic receives the same standardized inputs through its separate value branches plus a joint `1536 → 128 → 64 → 1` MLP residual, allowing value predictions to depend on current-scene and motion interactions. Hidden linear layers use orthogonal initialization.

For a fresh student policy, calibration measures the actual normalized student outputs on training-stream screenshots in `models/distillation/frames.pt`; held-out stream 3 is excluded. Mean and standard deviation (floor 0.001) initialize the policy buffers and count as 1,024 prior observations. Current features subtract the training mean; deltas use the same fixed per-coordinate scale without mean subtraction. Both are clipped to ±10. Delta scaling is not an empirical temporal-variance estimate. Teacher MLP policies start from identity statistics; a fresh run seeds them from its first rollout before the first update (zero-initialized output layers make this output-preserving). After every PPO update, the rollout's features are merged into the running mean/std (`running-rollout-v1`), never between an action and its update, so stored log-probabilities stay valid. This matters because EmbeddingGemma frame embeddings have mean pairwise cosine 0.986: unstandardized, the network sees nearly constant inputs. This calibration addresses weak frame variation rather than adding game metadata or changing rewards.


An existing checkpoint resumes its saved architecture; older checkpoints are interpreted as `legacy`. Resume/play/evaluation reuse saved calibration without requiring the frame dataset. An explicit incompatible architecture override is rejected. To use the nonlinear policy with a legacy run, start a fresh checkpoint. Recovery-only batches leave all actor branches unchanged, including Adam momentum and weight decay; only value heads update.

The previous legacy student's held-out action probabilities had mean across-frame standard deviation 0.00032 and value standard deviation 0.00120 over 255 screenshots, despite substantial scene changes. Its final 20 logged updates averaged explained variance −0.456. These motivate the architecture experiment; they do not establish that the new network improves game completion.

The fresh MLP diagnostic completed 10,000 emulator frames, 833 decisions and 52 PPO updates: 19 deaths, zero clears and three heuristic gap detections. Both zero-initialized new output layers acquired nonzero weights. On the same 255 held-out student embeddings with previous-frame context reset at episode boundaries, mean across-frame action-probability standard deviation was 0.04466 for the MLP versus 0.000348 for the legacy checkpoint. This measures input sensitivity, not correct action selection; the checkpoints have different training histories. With one CPU thread, 20 warmups and 100 timed action calls, policy-only median/p95 latency was 0.131/0.136 ms for the MLP versus 0.081/0.093 ms for legacy. Runtime encoder median/p95 was 2.017/2.708 ms. `docs/mlp-network-diagnostic.json` contains the measured report. Improved gameplay has not been established.
