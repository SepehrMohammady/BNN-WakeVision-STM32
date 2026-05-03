# Binarized Wake-Up of Conversational Agents on STM32H7B3I-DK

Companion repository for the IEEE COINS 2026 paper *Binarized Wake-Up of Conversational Agents on an Industry-Grade High-Performance Microcontroller* (Paper ID 2836).

Two complementary binary deployment pipelines for the **WakeVision** person-detection benchmark on a single STM32H7B3I-DK target (Cortex-M7 @ 280 MHz, 1.4 MB SRAM, 2 MB Flash):

1. **NAS-BNN → LCE TFLite → TFLM firmware** — neural-architecture search over a binary supernet, exported via Larq Compute Engine, executed by TensorFlow Lite Micro with custom `LceBconv2d` kernels.
2. **Samy 4-bit → Larq binarization → CBin-NN C export** — model-centric binarization of the official WakeVision Samy reference, converted to standalone bit-packed C with the CBin-NN inference engine.

Headline numbers on STM32H7B3I-DK (see paper Table III):

| Path | Latency | Power | Energy | RAM | Flash | Acc. |
|------|--------:|------:|-------:|----:|------:|-----:|
| NAS-BNN Key 4 (TFLM+LCE, 32×32) | 50.29 ms | 60 mW | 3.02 mJ | 110 KiB | 418 KiB | 80.41% |
| Binary Samy (CBin-NN, 80×80) | **30.5 ms** | 80 mW | **2.4 mJ** | **9.38 KiB** | **7.03 KiB** | 73.35% |

## Repository layout

```
.
├── paper/                 # IEEE COINS 2026 submission (LaTeX source + PDF + figures)
├── nasbnn/                # NAS-BNN supernet, search, fine-tuning, LCE export
├── checkpoints/           # Fine-tuned NAS-BNN PyTorch checkpoints (Keys 3–6)
├── search_results/        # Diversity-aware Pareto search artifacts
├── tflite_models/         # LCE TFLite + int8 reference TFLite per key
├── stm32_firmware/        # STM32CubeIDE project: TFLM + LCE binary inference
└── samy_cbinnn/           # Samy 4-bit → binary retraining + CBin-NN C path
```

## Downloading checkpoints

Fine-tuned NAS-BNN checkpoints (~1.9 GB total) are too large for regular git. Download from the [GitHub Releases page](../../releases) of this repository:

```
checkpoints/
  nasbnn_key3_best_ep29_acc80.31_f10.7895.pth.tar
  nasbnn_key4_best_ep29_acc80.41_f10.7934.pth.tar
  nasbnn_key5_best_ep23_acc80.61_f10.7930.pth.tar
  nasbnn_key6_best_ep27_acc80.88_f10.7937.pth.tar
```

Place them in `checkpoints/` before running evaluation scripts.

## Pipeline 1 — NAS-BNN

* `nasbnn/` — supernet, search (`search.py`), fine-tuning (`train.py`, `train_single.py`), LCE export (`build_lce_tflite.py`), evaluation (`eval_*.py`).
* `checkpoints/` — best-epoch fine-tuned weights for Keys 3, 4, 5, 6 (Pareto front).
* `tflite_models/` — `nasbnn_keyN_lce.tflite` (binary, deployed) and `nasbnn_keyN_int8_static.tflite` (reference).
* `stm32_firmware/` — STM32CubeIDE project. Native 32×32 deployment in `Core/Src/benchmark_main.cpp`. Tile model embedded as `Core/Inc/model_data.h`. DWT-based latency, UART-printed mean/min/max.

Build firmware with STM32CubeIDE 1.13+. External libraries (TFLM + LCE compute-engine + flatbuffers + gemmlowp) are not vendored here; see paper §III-C for build flags.

## Pipeline 2 — Samy + CBin-NN

* `samy_cbinnn/Samy_binary.ipynb` — Larq retraining of the WakeVision Samy reference under {−1,+1} weight/activation constraints (PReLU, Adam, augmentation; plateaus at 73.35%).
* `samy_cbinnn/samy_bnn.h5` — final binarized weights.
* `samy_cbinnn/samy_latent_weights.h5` — latent (real-valued) training-time weights.

C export uses the CBin-NN engine ([Sakr et al., *Electronics*, 2024](https://doi.org/10.3390/electronics13091624)) — bit-packed weights, ping-pong activation buffers, static memory schedule.

## Reproducing paper results

* **Search**: `nasbnn/search.py` — 250 archs, pop 50, 10 generations, 70/20/10 diversity-aware Pareto selection.
* **Fine-tune**: `nasbnn/train.py` — 30 epochs, batch 256, LR 1e-4, KD with smallest+largest sub-nets, label smoothing 0.1.
* **Eval**: `nasbnn/eval_finetuned.py` — full 55,762 WakeVision test split.
* **STM32 latency**: flash `stm32_firmware/`, read UART @ 115200 — DWT counter, 10 timed inferences after 2 warm-ups.
* **Power**: digital multimeter in series with board supply; report active − idle.

## Citation

```bibtex
@inproceedings{anon2026binarizedwakeup,
  title     = {Binarized Wake-Up of Conversational Agents on an Industry-Grade High-Performance Microcontroller},
  booktitle = {Proc.\ IEEE Int.\ Conf.\ Omni-Layer Intelligent Systems (COINS)},
  year      = {2026}
}
```

## License

See `nasbnn/LICENSE`. Third-party components retain their respective licenses (NAS-BNN: VDIGPKU; Larq Compute Engine: Plumerai; CBin-NN: ELIOS lab; TensorFlow Lite Micro: Apache 2.0).
