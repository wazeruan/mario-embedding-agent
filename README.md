# Mario EmbeddingGemma-2 PPO agent

Fresh rewrite in `/Users/weiciruan/Documents/mario-embedding-agent`.

A frozen image/text **google/embeddinggemma-2** encoder maps each RGB game screenshot to a normalized 768-dimensional vector. Action descriptions are encoded once and cached. A trainable linear actor starts with those vectors as its weights, then learns independent action logits through PPO. A categorical softmax samples actions during training; play uses argmax. A separate nonlinear critic estimates the discounted return: LayerNorm → 256 → GELU → 128 → GELU → value. The encoder and cached action prototypes remain frozen. There is no LoRA or generated answer in this rewrite.

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
```

Sync pauses emulator advancement during encoding, so each decision uses the current screenshot. Its wall-clock game rate is slower; this mode helps inspect mistakes without inference staleness. `--fps 30` slows emulator pacing in either mode. There is no default time or frame cutoff. Close the window, press Escape or Ctrl+C to stop and save. `--headless --total-frames 600` is an explicit finite diagnostic. Models must be prepared before train/play. Checkpoints from the previous LoRA implementation are incompatible.

## Actions and objective

There are 18 actions: idle, walk/run right, walk/run left, down, up, fire, and short/long jumps vertically, right, running right, left, running left. They are button-duration macros, not guaranteed successful maneuvers; jump height also depends on game physics and current movement.

Rewards are +1 for each policy decision whose RGB observations show Mario advancing right, +1 for a visually detected level transition, and −1000 for death. A confirmed jump over a gap adds a separate +1 to that decision. There is no per-decision time penalty. Death and level-clear outcomes override movement and gap rewards. Enemy elimination receives no bonus. PPO divides these rewards by 1000; its critic predicts scaled returns. The default rollout is 16 decisions, two epochs, minibatches of eight, AdamW at 0.0003, discount 0.99 and GAE lambda 0.95. Each macro costs one decision regardless of duration, so this objective does not directly minimize elapsed emulator frames.

Only RGB screenshots reach the policy and success detector. The emulator adapter exposes the permitted death-only `terminated` flag and suppresses game rewards/info and RAM-based animation skipping. The policy never reads score, position, lives, time, or other game metadata. Death frames are saved before reset. Level success is detected by a stable change in the rendered WORLD number over three frames; the game continues into the next level without reset. This detects intermediate level transitions, not final-game victory. Enemy elimination receives no reward.

## Training, logs and continuation

The game window shows action, inference latency, observation age, rollout fill, updates, deaths, clears, raw/scaled reward, losses, entropy, explained variance and top action probabilities. The smaller image is the screenshot actually encoded for the current decision.

`runs/train.jsonl` records decisions, PPO updates and outcomes; `runs/death-*.png` and `runs/clear-*.png` retain outcome screenshots. `checkpoints/latest.pt` stores actor, critic, optimizer, RNG, counters and the current episode's controller commands. Resume deterministically replays that episode from its reset seed using only screenshots and controller commands. An unfinished rollout/action is discarded at stop and a fresh decision begins on resume. The seed saved in the checkpoint takes precedence over `--seed` on resume. FPS, mode and rollout can be changed when continuing. Replay time grows with episode length.

Cached RGB frames avoid repeated encoder work only when the frame is byte-identical. PPO reuses detached features and never reruns the encoder for gradient epochs. Model latency and cache-hit latency are separate measurements. Faster training and pretrained embeddings do not guarantee level completion; gameplay improvement requires evaluation across episodes.

### Visual progress and gap rewards

Rightward progress is estimated from Mario sprite movement plus horizontal scenery scroll between RGB frames. A decision gets +1 when its observed net world displacement is more than one game pixel to the right; button presses alone do not earn reward. If Mario is not detected in a frame, the movement baseline resets; later observations cannot carry missing-frame displacement into a different policy action. Ambiguous scenery matches contribute zero estimated scroll; detectable Mario screen movement is still counted. Gap crossings are recognized from a sequence of RGB frames: a grounded approach to a visible hole at the ground row, airborne passage, then three grounded frames beyond the far edge. A crossing is counted once while its gap remains tracked, and its +1 is added to the rightward movement reward for that policy decision. This conservative detector supports the original small-Mario, blue-sky overworld palette and ground height; other palettes, large/fire Mario, obscured sprites or gaps leaving the viewport may be missed. It is a heuristic, not general scene understanding. `runs/gap-*.png` records detected landings and `gap_crossing` log events report the +1 bonus. The UI shows the gap count.

Continuing a checkpoint created before either shaping reward was added preserves weights and optimizer but starts fresh rollouts with the current reward objective. This migration is explicitly recorded in the `started` log event. Detector history is reconstructed during episode replay without awarding bonuses again.
