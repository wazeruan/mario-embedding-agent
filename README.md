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

The default visible realtime mode keeps the emulator running at a target of 60 FPS while one encoder worker processes screenshots. It has no inference backlog. Short jumps hold A for 4 emulator frames and release it for 2; long jumps hold A for 20 and release for 2. While waiting for the next decision, the released controller command continues, so inference cannot accidentally extend a short jump. Incoming observations can still be several frames old; the UI shows their age and exact input screenshot.

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

For an ordinary policy decision, movement reward is signed RGB-estimated world displacement: `clip(Δx / 8, −5, +5)`, with movements of at most one pixel treated as tracking noise. Moving forward 16 pixels earns +2; moving back 16 pixels earns −2, so a round trip earns approximately zero. The one-pixel deadband and per-decision clipping can make the result differ slightly when the same movement is split into many actions. A confirmed jump over a gap adds +1. Death gives −50, and a visually detected level transition gives +1; either terminal outcome overrides movement and gap rewards. Enemy elimination receives no reward. PPO divides rewards by 50. The default rollout is 16 decisions, two epochs, minibatches of eight, AdamW at 0.0003, discount 0.99, GAE lambda 0.95, and entropy coefficient 0.001. The −50 death penalty is an initial tuning choice after repeated deaths under the prior −10 setting; per-action reward events are now logged so it can be calibrated from measured movement and survival intervals. Signed rewards cancel equal forward and backward movement, avoiding reward from simply oscillating; the deadband and clipping caveat above applies. Gap bonuses occur once per crossing. Each completed policy action also writes a `decision_result` JSONL event with its action, duration, image-estimated displacement, gap bonus, raw/scaled reward, and outcome.

Only RGB screenshots reach the policy and success detector. The emulator adapter exposes the permitted death-only `terminated` flag and suppresses game rewards/info and RAM-based animation skipping. The policy never reads score, position, lives, time, or other game metadata. Death frames are saved before reset. Level success is detected by a stable change in the rendered WORLD number over three frames; the game continues into the next level without reset. This detects intermediate level transitions, not final-game victory. Enemy elimination receives no reward.

## Training, logs and continuation

The game window shows action, inference latency, observation age, rollout fill, updates, deaths, clears, raw/scaled reward, losses, entropy, explained variance and top action probabilities. The smaller image is the screenshot actually encoded for the current decision.

`runs/train.jsonl` records decisions, PPO updates and outcomes; `runs/death-*.png` and `runs/clear-*.png` retain outcome screenshots. `checkpoints/latest.pt` stores actor, critic, temporal heads, optimizer, RNG, counters, prior embedding and the current episode's controller commands. Resume deterministically replays that episode from its reset seed using only screenshots and controller commands. An unfinished rollout/action is discarded at stop and a fresh decision begins on resume. The seed saved in the checkpoint takes precedence over `--seed` on resume. FPS, mode and rollout can be changed when continuing. Replay time grows with episode length.

Cached RGB frames avoid repeated encoder work only when the frame is byte-identical. PPO reuses detached features and never reruns the encoder for gradient epochs. Model latency and cache-hit latency are separate measurements. Faster training and pretrained embeddings do not guarantee level completion; gameplay improvement requires evaluation across episodes.

### Visual progress and gap rewards

Rightward progress is estimated from Mario sprite movement plus horizontal scenery scroll between RGB frames. Button presses alone do not earn reward. If Mario is not detected in a frame, the movement baseline resets; later observations cannot carry missing-frame displacement into a different policy action. Ambiguous scenery matches contribute zero estimated scroll; detectable Mario screen movement is still counted. Gap crossings are recognized from a sequence of RGB frames: a grounded approach to a visible hole at the ground row, airborne passage, then three grounded frames beyond the far edge. A crossing is counted once while its gap remains tracked, and its +1 is added to the rightward movement reward for that policy decision. This conservative detector supports the original small-Mario, blue-sky overworld palette and ground height; other palettes, large/fire Mario, obscured sprites or gaps leaving the viewport may be missed. It is a heuristic, not general scene understanding. `runs/gap-*.png` records detected landings and `gap_crossing` log events report the +1 bonus. The UI shows the gap count.

Resuming a checkpoint from an earlier reward objective preserves actor weights, critic feature layers, optimizer state for the actor, and episode counters; it migrates the reward scale and death penalty, applies the left-action prior and lower entropy coefficient, and starts fresh rollouts. Because value targets change with the reward objective, migration zeros both critic output layers and clears their Adam moments while retaining learned critic features. The `started` event records reward and temporal-context migrations. Detector history is reconstructed during episode replay without awarding bonuses again.
