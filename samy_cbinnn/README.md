# Samy → Binary → CBin-NN

Model-centric binary deployment path of the paper. Starts from the WakeVision Challenge **Samy** 4-bit model (80×80 input, 5.58 M-MACs, 79.9% accuracy) and produces a standalone bit-packed C inference for STM32H7B3I-DK.

## Files

| File | Purpose |
|------|---------|
| `Samy_binary.ipynb` | Larq retraining notebook: {−1,+1} weight + activation constraints, PReLU activations, Adam (in place of BOP), data augmentation, hyperparameter sweeps. |
| `samy_latent_weights.h5` | Real-valued latent weights from training (full precision shadow used by Larq's STE). |
| `samy_bnn.h5` | Final binarized weights, ready for export. |

## Result on STM32H7B3I-DK

| Metric | Value |
|--------|------:|
| Accuracy (WakeVision test, 55,762 samples) | 73.35% |
| Latency | 30.5 ms |
| Power (active − idle) | 80 mW |
| Energy / inference | 2.4 mJ |
| Peak RAM | 9.38 KiB |
| Flash | 7.03 KiB |

vs. original 4-bit Samy: same 5.58 M-MAC budget, **8× smaller Flash**, **~3× smaller RAM**, ~20% lower latency, at a 6.55-pp accuracy cost.

## CBin-NN export

C code generation uses the public CBin-NN engine:

> F. Sakr et al., *CBin-NN: An Inference Engine for Binarized Neural Networks*, Electronics 13:1624, 2024. <https://doi.org/10.3390/electronics13091624>

CBin-NN packs binary weights cross-channel (32 weights / 32-bit word), generates XNOR-popcount Conv kernels, and emits a static activation schedule with two ping-pong buffers — only two layer-sized activations resident at once.

The generated C sources are not vendored here (separate license / upstream toolchain). To regenerate: load `samy_bnn.h5` in CBin-NN, run the C-export pass targeting Cortex-M7, build into the STM32 firmware as a single `.c` + `.h` pair.

## Reproduce binarization

Open `Samy_binary.ipynb`. Requires:

```
larq>=0.13
tensorflow>=2.10
numpy
```

Train on the WakeVision training split (5.76 M images). The notebook documents the hyperparameter sweep that converged at 73.35% — accuracy plateaus past ~30 epochs.
