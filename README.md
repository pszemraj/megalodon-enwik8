# megalodon-enwik8

Minimal example demonstrating **[MEGALODON](https://arxiv.org/abs/2404.08801)** outperforms Llama-style Transformers on character-level language modeling after the **same number of training steps**. A companion repo to [megalodon-hf](https://github.com/pszemraj/megalodon-hf).

## Results

Similarly-sized architectures trained on enwik8 for 1200 steps (_10% of the dataset_):

| Model         | Parameters | Val Loss @ 1100 | BPC      | VRAM  | Time   |
| ------------- | ---------- | --------------- | -------- | ----- | ------ |
| **Megalodon** | 11.3M      | **1.451**       | **2.09** | 12 GB | 8m 09s |
| Llama         | 12.5M      | 1.542           | 2.22     | 7 GB  | 3m 07s |

Megalodon achieves **5.9% lower loss** with **10% fewer parameters**, but at higher compute cost (~2.6x slower, ~1.7x VRAM).

- This is expected: torch lacks native support for Megalodon's complex-valued EMA operators (_and [megalodon-hf](https://github.com/pszemraj/megalodon-hf) explicitly eschews complex-value CUDA kernels a la the upstream_), while Transformers benefit from years of kernel optimization.
- See [RESULTS.md](RESULTS.md) for experimental details and mitigation paths w.r.t. speed and memory.

This repo exists to demonstrate that Megalodon _can_ outperform Transformers in a controlled setting, and as such focuses on correctness & readability over speed.

## Quick Start

```bash
# Install
pip install -e .

# Train Megalodon (primary)
python train.py --config configs/megalodon_multichunk_512.yaml

# Train Llama baseline
python train.py --config configs/llama_512.yaml
```

## What This Repo Is

A **sanity check / MWE** showing [megalodon-hf](https://github.com/pszemraj/megalodon-hf) works. It demonstrates that Megalodon learns to generate coherent text at small scale and beats an equivalent Transformer.

## What This Repo Is NOT

- A pretraining framework
- A comprehensive benchmark suite
- Production-ready training code

The modeling code [can be found here](https://github.com/pszemraj/megalodon-hf).

## Requirements

- PyTorch >= 2.9.0 with **bfloat16 support** (Ampere+ GPU or modern CPU)
- **float16 is NOT supported** due to numerical overflow in complex EMA

Dependencies are installed via `pip install -e .`

## Device Support

| Device        | Status       | Notes                                                    |
| ------------- | ------------ | -------------------------------------------------------- |
| NVIDIA GPU    | Best         | flash attention, SDPA for supported ops                  |
| Apple Silicon | Untested[^1] | MPS backend with autocast, template works for `llama.py` |
| CPU           | Works        | Slow; use for testing only                               |

Override device with `FORCE_DEVICE=cuda`, `FORCE_DEVICE=mps`, or `FORCE_DEVICE=cpu`.

[^1]: I do not have access to Apple Silicon hardware for testing megalodon-specific training. The repo template did test `llama.py` on MPS and confirmed it works. Please open an issue/PR if you try this out.

## Structure

```text
megalodon-enwik8/
├── megalodon_enwik8/    # Model implementations
│   ├── megalodon.py     # MegalodonLM wrapper
│   ├── llama.py         # Llama baseline
│   └── utils.py         # Sampling & device helpers
├── configs/
│   ├── megalodon_multichunk_512.yaml  # Primary config
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
