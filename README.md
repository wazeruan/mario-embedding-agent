# Mario embedding PPO agent

A fresh implementation in Documents. RGB screenshots are encoded by a frozen local copy of Google's EmbeddingGemma-2. The actor is a trainable linear 768→18 action projection initialized from pre-encoded action descriptions. Softmax defines the action distribution. A separate LayerNorm→256→GELU→128→GELU→1 critic learns scaled returns. There is no LoRA or trainable encoder in this rewrite.

The action embeddings initialize the classifier weights; they are retained as an immutable reference. PPO can subsequently change classifier weights freely. Pretrained semantic similarity is an initialization, not evidence of game-playing competence.

## Environment and model

```bash
uv sync --frozen
uv run python -c 'from mario_embedding_agent.encoder import download_model; print(download_model())'
```

The model is pinned to revision `914f7f89142e33e77833254d9c9b90c3cef7303b` and stored in `models/embeddinggemma-2`. Image/text inference disables the audio encoder. MPS/CUDA use bfloat16; CPU uses float32. Default image budget is 70 tokens and output dimension is 768. See the [official model card](https://huggingface.co/google/embeddinggemma-2) for input/precision guidance. The dependency versions are locked in `uv.lock`; `.venv` belongs only to this project.

## Actions and learning

There are 18 button/duration actions: idle, walking/running in either direction, up, down, fire, and short/long jumps vertically or while moving/running left/right. Short jump holds A for 4 emulator frames, long jump for 20; both force 2 subsequent frames with A released while retaining directional/run buttons. Fire also releases B after its tap. This prevents repeated selection of a short jump from silently becoming a continuous hold.

PPO uses 16 macro decisions per rollout, two epochs, minibatches of eight, clipped policy and value losses, entropy regularization, gradient clipping, and GAE. Cached frozen 768-dimensional features are reused in every PPO epoch; optimization never reruns the encoder. Actor and critic train on CPU, leaving the accelerator to the image encoder.

The reward objective is −1 per ordinary macro decision, +1 for a visually confirmed next-level transition, and −1000 for death; terminal outcomes override the decision cost. All rewards are divided by 1000 before learning. Gamma=.99 and GAE lambda=.95 are per decision, not per emulator frame: long and short macro actions have the same decision cost/discount despite different frame durations. The objective therefore favors fewer decisions, rather than directly minimizing wall-clock or simulator time.

Checkpoints atomically save actor, critic, frozen action vectors, optimizer, RNG, settings, and replay state. The previous project's LoRA/seven-action checkpoints are incompatible with this direct-actor/eighteen-action architecture and are not overwritten. A rewritten model starts fresh.
