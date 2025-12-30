# megalodon-enwik8

Minimal example demonstrating **MEGALODON** outperforms Llama-style Transformers on character-level language modeling.

## Results

| Model         | Parameters | Val Loss @ 1100 steps | BPC      |
| ------------- | ---------- | --------------------- | -------- |
| **Megalodon** | 11.3M      | **1.453**             | **2.10** |
| Llama         | 12.5M      | 1.534                 | 2.21     |

Megalodon achieves **5.3% lower loss** with **10% fewer parameters**.

> See [RESULTS.md](RESULTS.md) for detailed experimental results.

## Quick Start

```bash
# Install
pip install -e .

# Train Megalodon (primary)
python train.py --config configs/mega_multichunk_512.yaml

# Train Llama baseline
python train.py --config configs/llama_512.yaml
```

## What This Repo Is

A **sanity check / MWE** showing [megalodon-hf](https://github.com/pszemraj/megalodon-hf) works. It demonstrates that Megalodon learns to generate coherent text at small scale and beats an equivalent Transformer.

## What This Repo Is NOT

- A pretraining framework
- A comprehensive benchmark suite
- Production-ready training code

The real code is at: <https://github.com/pszemraj/megalodon-hf>

## Requirements

- PyTorch >= 2.9.0 with **bfloat16 support** (Ampere+ GPU or modern CPU)
- **float16 is NOT supported** due to numerical overflow in complex EMA

Dependencies are installed via `pip install -e .`

## Device Support

| Device        | Status | Notes                             |
| ------------- | ------ | --------------------------------- |
| NVIDIA GPU    | Best   | Fused optimizer & flash attention |
| Apple Silicon | Good   | MPS backend with autocast         |
| CPU           | Works  | Slow; use for testing only        |

Override device with `FORCE_DEVICE=cuda`, `FORCE_DEVICE=mps`, or `FORCE_DEVICE=cpu`.

## Structure

```text
megalodon-enwik8/
├── megalodon_enwik8/    # Model implementations
│   ├── megalodon.py     # MegalodonLM wrapper
│   ├── llama.py         # Llama baseline
│   └── utils.py         # Sampling & device helpers
├── configs/
│   ├── mega_multichunk_512.yaml  # Primary Megalodon config
│   ├── llama_512.yaml # Matched Llama baseline
│   └── test.yaml                 # Quick smoke test
├── data/
│   └── enwik8.gz        # Character-level dataset
├── train.py             # Training script
├── inference.py         # Inference utility
└── RESULTS.md           # Experimental results
```

## License

MIT (this repo) / Apache-2.0 ([megalodon-hf](https://github.com/pszemraj/megalodon-hf))

## References

Megalodon architecture:

```bibtex
@misc{ma2024megalodon,
      title={Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length},
      author={Xuezhe Ma and Xiaomeng Yang and Wenhan Xiong and Beidi Chen and Lili Yu and Hao Zhang and Jonathan May and Luke Zettlemoyer and Omer Levy and Chunting Zhou},
      year={2024},
      eprint={2404.08801},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Llama baseline:

```bibtex
@misc{touvron2023llama2,
      title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
      author={Hugo Touvron and others},
      year={2023},
      eprint={2307.09288},
      archivePrefix={arXiv}
}
```
