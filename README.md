# Mario EmbeddingGemma-2 PPO agent

Fresh rewrite in `/Users/weiciruan/Documents/mario-embedding-agent`.

A frozen image/text **google/embeddinggemma-2** encoder maps each RGB game screenshot to a normalized 768-dimensional vector. Action descriptions are encoded once and cached. The actor starts as a linear projection initialized from those action vectors; a second, zero-initialized linear head learns a residual from the difference between the current and previous decision embeddings. The critic combines its nonlinear current-frame value estimate with a zero-initialized MLP residual on the same embedding difference. PPO trains both residuals while the image encoder and action prototypes stay frozen. A categorical softmax samples actions during training; play and evaluation use argmax. No game RAM or numerical game state enters the policy.

Temporal context spans policy decisions rather than every emulator frame: it can reveal visual motion or a scene change that a single screenshot cannot. The exact prior embedding is stored with each PPO transition so training recomputes the same action probability, and it is checkpointed for continuation. Death and detected WORLD transitions clear this context. Existing direct-actor checkpoints migrate with zero-initialized temporal heads, so their starting behavior is preserved while PPO learns whether the new signal helps.

The pinned [official encoder](https://huggingface.co/google/embeddinggemma-2) revision is `914f7f89142e33e77833254d9c9b90c3cef7303b`; image budget 70 soft tokens, audio disabled. Model files and action vectors are local under `models/` and ignored by Git.

## Run

```sh
cd /Users/weiciruan/Documents/mario-embedding-agent
uv sync
uv run mario-agent prepare
uv run mario-agent train
```

The default visible realtime mode keeps the emulator running at a target of 60 FPS while one encoder worker processes screenshots. It has no inference backlog. Short jumps hold A for 4 emulator frames and release it for 2; long jumps hold A for 20 and release for 2. While waiting for the next decision, all controller buttons are released after the macro, so inference cannot accidentally extend a short jump. Incoming observations can still be several frames old; the UI shows their age and exact input screenshot.

```sh
uv run mario-agent train --resume checkpoints/latest.pt
uv run mario-agent play --checkpoint checkpoints/latest.pt
uv run mario-agent train --mode sync
uv run mario-agent evaluate --checkpoint checkpoints/latest.pt --episodes 10 --seed 0 --mode sync
```

Sync pauses emulator advancement during encoding, so each decision uses the current screenshot. Its wall-clock game rate is slower; this mode helps inspect mistakes without inference staleness. Evaluation defaults to sync so repeated checkpoint/seed comparisons do not depend on encoder thread timing. `--fps 30` slows emulator pacing in either mode. There is no default time or frame cutoff. Close the window, press Escape or Ctrl+C to stop and save. `--headless --total-frames 600` is an explicit finite diagnostic. Models must be prepared before train/play/evaluate. Direct-actor PPO v1 checkpoints migrate automatically; checkpoints from the earlier LoRA implementation are incompatible.

`evaluate` loads a checkpoint's policy and runs deterministic argmax actions without updating or saving weights. It reports each death-ended attempt plus aggregate frames, decisions, screenshot-derived reward/progress, detected gap crossings and world transitions. Seeds advance only after a visually observed death; there is no per-attempt time/frame cutoff. This is a frozen-policy comparison, not proof of final-game completion; the existing clear detector sees intermediate WORLD transitions.

## Actions and objective

There are 18 actions: idle, walk/run right, walk/run left, down, up, fire, and short/long jumps vertically, right, running right, left, running left. They are button-duration macros, not guaranteed successful maneuvers; jump height also depends on game physics and current movement. The policy applies a −3 logit prior to actions containing left, reducing their baseline softmax mass; frame-conditioned logits can still select left when their evidence overcomes that prior. Training uses an entropy coefficient of 0.001.

To break a visual stall, the controller counts consecutive completed actions whose RGB-estimated rightward displacement is at most one pixel and which did not cross a detected gap. After four, it biases the policy through a recovery sequence: hold `down` for 24 frames to try entering a pipe, back away (`run_left`), then jump/run right. One brief movement burst no longer exits recovery: it takes two progress-bearing actions in a four-action window with positive net displacement, or a detected gap crossing, to confirm escape. Death or a rendered level transition resets recovery. This uses rendered-frame motion only. The recovery stage and progress window are saved in checkpoints. The temporary action bias is stored with that PPO transition and reapplied during PPO updates, so the sampled-action probability remains consistent with the probability used for training. The UI and JSONL logs report the stagnation count, recovery action, and accumulated escape evidence.

For an ordinary policy decision, movement reward pays for newly reached visual ground: `clip(new_frontier_pixels / 32, 0, 5)`. A stateful frontier integrates Mario sprite movement and scenery scroll from RGB frames. Advancing 32 pixels beyond the previous frontier earns +1; retreating or retracing already visited ground earns zero, so a repeated left/right cycle cannot farm reward. There is no per-decision penalty or enemy-elimination reward. A confirmed gap crossing adds +1. Death gives −50 and a detected WORLD transition gives +50; either terminal outcome overrides movement and gap rewards, and death takes precedence. The frontier resets on death or WORLD transition. This remains a visual heuristic: tracking errors can still shift the frontier, and missing sprites contribute no guessed motion.

PPO divides rewards by 50 and retains the requested 16-decision rollout. Each rollout receives up to four epochs with minibatches of 16, AdamW at 0.0003 with zero weight decay, clip 0.2, discount 0.99 and GAE lambda 0.95 per 20 emulator frames, and entropy coefficient 0.001. Each transition stores its actual elapsed frames, including time waiting for the next embedding. Discounts and GAE traces use `gamma ** (frames / 20)` and `lambda ** (frames / 20)`, so short and long actions use the same game-time horizon. Actor loss and entropy use only unassisted decisions; recovery decisions still train the critic. A KL guard stops the update when estimated policy divergence exceeds 1.5 times the 0.02 target, following the [PPO early-stopping practice](https://spinningup.openai.com/en/latest/algorithms/ppo.html). This protects update size; it does not establish improved gameplay.

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

The small CNN learns the frozen teacher's 768-dimensional normalized image embeddings from real Mario screenshots. It predicts standardized embedding residuals using statistics computed only on training frames. This prevents dominant shared embedding directions from disguising a constant-output student. Action text vectors remain pre-encoded by the teacher; the student accepts images only. PPO continues to train its actor/critic from detached student features; it does not update the encoder online.

```sh
uv run mario-agent prepare
uv run mario-distill collect --samples 1024
uv run mario-distill train --epochs 60
uv run mario-distill benchmark
uv run mario-agent train --student-encoder models/distillation/student.pt --checkpoint checkpoints/student-scratch.pt
```

Collection interleaves four independently seeded emulator streams, retains original RGB frames and teacher vectors, and resets each environment only after death. Exact duplicate frames are removed globally. Stream 3 is held out entirely, including all its episodes; this is a small pilot dataset rather than comprehensive gameplay coverage. Both training and benchmark reports compare the student with the training-mean embedding baseline. The benchmark measures uncached, warmed inference on the same held-out screenshots and writes `models/distillation/student.benchmark.json`; embedding agreement is not evidence of improved gameplay.

The default encoder remains EmbeddingGemma-2. Student inference defaults to CPU and does not load the teacher. Keep the prepared action cache. Start a fresh PPO checkpoint when switching encoders; resume requires the identical student artifact fingerprint. Re-distilling creates a new encoder identity. Increase data diversity before treating the pilot student as a replacement.

### Local pilot results (2026-10-08)

Seed 0 collection yielded 1,024 globally unique real screenshots across 25 stream/episode groups: 769 training and 255 held-out stream-3 frames. After 60 epochs, the best student checkpoint had held-out cosine 0.99492 and standardized RMSE 0.86749, versus 0.99346 and 0.97214 for the constant training-mean baseline (about 11% lower error). Pairwise-distance correlation was 0.597, so frame geometry is only partially preserved.

On 32 evenly spaced held-out screenshots, warmed uncached inference including preprocessing and CPU feature return measured median 0.735 ms / p95 0.778 ms for the CPU student, versus 162.907 ms / 186.460 ms for the MPS teacher. This is approximately 222× faster for the encoder on this machine; it does not imply 222× faster decisions or PPO. The student has 499,728 parameters and its artifact is about 2 MB. Benchmark-subset cosine was 0.99473, standardized RMSE 0.88020 versus the mean baseline's 0.96241, and pairwise-distance correlation 0.501. Action-prototype top-1 agreement was 100% on this small subset; this is retrieval agreement, not PPO action agreement or game success. Gameplay improvement has not been evaluated.
