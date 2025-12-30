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

## Parameter Counts

| Model     | Parameters |
| --------- | ---------- |
| Megalodon | 11,277,696 |
| Llama     | 12,489,792 |

Megalodon has ~10% fewer parameters than the matched Llama baseline.

## Final Metrics (seed=7)

| Model         | Val Loss @ 1100 | BPC      |
| ------------- | --------------- | -------- |
| **Megalodon** | **1.453**       | **2.10** |
| Llama         | 1.534           | 2.21     |

> BPC (bits per character) = val_loss / ln(2)

**Result**: Megalodon achieves **5.3% lower validation loss** than Llama with 10% fewer parameters.

## Training Curves

| Step | Megalodon Val Loss | Llama Val Loss |
| ---- | ------------------ | -------------- |
| 0    | 5.673              | 5.674          |
| 100  | 2.025              | 2.594          |
| 200  | 1.707              | 2.285          |
| 300  | 1.517              | 2.059          |
| 400  | 1.552              | 1.871          |
| 500  | 1.570              | 1.821          |
| 600  | 1.565              | 1.715          |
| 700  | 1.546              | 1.505          |
| 800  | 1.507              | 1.646          |
| 900  | 1.507              | 1.643          |
| 1000 | 1.446              | 1.665          |
| 1100 | 1.453              | 1.534          |

## Reproduction

```bash
# Train Megalodon
python train.py --config configs/mega_multichunk_512.yaml

# Train Llama
python train.py --config configs/llama_512.yaml
```

Checkpoints saved to `runs/megalodon/` and `runs/llama/`.
