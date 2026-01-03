# Experimental Results

> [!IMPORTANT]
> **Disclaimer**: This is a sanity check, not a rigorous benchmark. Both models use identical hyperparameters (LR, batch size, etc.) with no per-architecture tuning. The goal is to show Megalodon is viable and competitive, not to determine a winner. Take the specific numbers with a grain of salt; your mileage may vary with different configs, scales, or datasets.

## Setup

- **Dataset**: enwik8 (character-level, ~95M bytes used)
- **Sequence length**: 512
- **Chunk size**: 256 (Megalodon multi-chunk training)
- **Training steps**: 1200
- **Effective batch size**: 16 (batch_size=1, grad_accum=16)
- **Learning rate**: 4e-4
- **Precision**: bfloat16 (autocast)
- **Seed**: 7
- **Hardware**: NVIDIA GeForce RTX 5090
- **PyTorch**: 2.9.1+cu128

## Parameter Counts

| Model     | Parameters |
| --------- | ---------- |
| Megalodon | 11,277,696 |
| Llama     | 12,489,792 |

Megalodon has ~10% fewer parameters than the matched Llama baseline.

## Final Metrics (seed=7)

| Model         | Val Loss @ 1100 | BPC      | VRAM  | Time (1200 steps) |
| ------------- | --------------- | -------- | ----- | ----------------- |
| **Megalodon** | **1.451**       | **2.09** | 12 GB | 8m 09s            |
| Llama         | 1.542           | 2.22     | 7 GB  | 3m 07s            |

> BPC (bits per character) = val_loss / ln(2)

**Result**: Megalodon achieves **5.9% lower validation loss** than Llama with 10% fewer parameters, but at higher compute cost (~2.6x slower, ~1.7x more VRAM).

### On Training Efficiency

To be transparent: Megalodon is currently slower and more memory-intensive than the Llama baseline. At the same time, however, this specific training comparison is **heavily** in the Transformer "home field advantage" regime. Several factors contribute:

1. **More operations per layer**: Megalodon's architecture includes CEMA (_FFT-based convolution_), TimestepNorm (_streaming Welford statistics_), normalized attention, and gating, all of which are absent from standard Transformers.

2. **No fused kernels.** Every Megalodon-specific operation is composed from basic PyTorch primitives. Llama's attention benefits from highly optimized fused SDPA kernels throughout.

3. **Chunking overhead.** This config processes two 256-token chunks to demonstrate Megalodon's streaming capability, while Llama sees the full 512-token sequence at once.[^1]

4. **Complex tensor overhead.** CEMA uses complex-valued parameters and intermediates, which consume 2× the memory of equivalent real tensors.

5. **Lack of ecosystem support.** Transformer architectures have years of optimization (Flash Attention, compiler passes, quantization tools). Megalodon's operators are too new to have equivalent infrastructure.[^2]

[^1]: This is **by design** to demonstrate that Megalodon's key advantage, _subquadratic (linear)_ memory scaling with context length, functions properly in this implementation but adds overhead at short sequences. Megalodon is fundamentally designed for long contexts (_tens of thousands+ tokens_) where Transformers struggle.
[^2]:`torch.compile` cannot even trace through complex tensor backward passes yet ([pytorch/pytorch#125718](https://github.com/pytorch/pytorch/issues/125718)).

**Paths to parity:**

- Custom Triton/CUDA kernels for CEMA and TimestepNorm (what the original paper used)
- JAX port (XLA natively supports compiled complex autodiff)
- Upstream PyTorch complex tensor compiler support (in progress)

**This repo prioritizes correctness and readability over speed**. For production use, kernel optimization could yield 2-5x speedups.

## Training Curves

| Step | Megalodon Val Loss | Llama Val Loss |
| ---- | ------------------ | -------------- |
| 0    | 5.673              | 5.676          |
| 100  | 2.026              | 2.604          |
| 200  | 1.703              | 2.287          |
| 300  | 1.515              | 2.061          |
| 400  | 1.551              | 1.870          |
| 500  | 1.570              | 1.819          |
| 600  | 1.566              | 1.712          |
| 700  | 1.545              | 1.504          |
| 800  | 1.505              | 1.642          |
| 900  | 1.507              | 1.647          |
| 1000 | 1.447              | 1.665          |
| 1100 | 1.451              | 1.542          |

## Reproduction

```bash
# Train Megalodon
python train.py --config configs/megalodon_multichunk_512.yaml

# Train Llama
python train.py --config configs/llama_512.yaml
```

Checkpoints saved to `runs/megalodon/` and `runs/llama/`.
