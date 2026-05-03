"""
eval_all_epochs_full.py
=======================
Evaluates ALL 120 epoch checkpoints (keys 3-6, 30 epochs each) on the
WakeVision FULL **test** set.  Computes Top-1 accuracy, binary F1
(positive class = person_present), precision, and recall for every
epoch of every key.

Outputs (all under OUT_DIR):
  all_epochs_results.csv          – master table for all 120 runs
  key{K}_epoch_results.csv        – per-key table
  best_checkpoints/               – best epoch checkpoint per key (by F1)
  eval_summary.log                – human-readable summary

Efficient design: the static model is built ONCE per key; only the
state-dict is swapped for each epoch (to_static is called once to
register weight_s / bias_s parameters, subsequent epochs reload the
baked static weights directly from the checkpoint state-dict).
"""

import os
import re
import csv
import shutil
import logging
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import tuple2cand

# ---------------------------------------------------------------------------
# Configuration — edit these if paths differ
# ---------------------------------------------------------------------------
KEYS        = [3, 4, 5, 6]
BASE_DIR    = "work_dirs/wakevision_nasbnn_FULLEXP_run"
SEARCH_INFO = "work_dirs/wakevision_nasbnn_LARGEXP_run/search/info.pth.tar"
DATA_DIR    = "data/WakeVision_Full"
ARCH_NAME   = "superbnn_wakevision_large"
IMG_SIZE    = 128
BATCH_SIZE  = 128
WORKERS     = 0
GPU         = 0          # set None to force CPU
OUT_DIR     = os.path.join(BASE_DIR, "full_eval_results")
BEST_DIR    = os.path.join(OUT_DIR, "best_checkpoints")
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("eval_full")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_test_loader(data_dir: str, img_size: int,
                      batch_size: int, workers: int,
                      pin_memory: bool):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    testdir = os.path.join(data_dir, "test")
    if not os.path.isdir(testdir):
        raise FileNotFoundError(f"Test directory not found: {testdir}")
    dataset = datasets.ImageFolder(testdir, val_transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory)
    return loader, dataset.classes   # classes[0]=no_person_present, [1]=person_present


def build_static_model(arch_key: int, search_results: dict,
                       device: torch.device, img_size: int):
    """Instantiate model, call to_static() once with a dummy input."""
    arch_tuple = search_results["pareto_global"][arch_key]
    arch_cand  = tuple2cand(arch_tuple)
    model = models.__dict__[ARCH_NAME](sub_path=arch_cand, img_size=img_size)
    model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        model.to_static(dummy)
    return model


def load_epoch_weights(model: torch.nn.Module,
                       ckpt_path: str,
                       device: torch.device) -> float:
    """Load state-dict from checkpoint; return stored val_acc1."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys after load: {missing[:5]} ...")
    model.to(device)
    model.eval()
    return float(ck.get("val_acc1", 0.0))


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device):
    """
    Returns (top1_acc %, binary_f1, precision, recall).
    Positive class = 1 (person_present, alphabetically second).
    """
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs, _ = model(images, model.sub_path)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    n = len(labels)

    acc = (preds == labels).sum().item() / n * 100.0

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return acc, f1, precision, recall, tp, fp, fn, tn


def main():
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    log_path = os.path.join(OUT_DIR, "eval_summary.log")
    logger   = setup_logger(log_path)

    device = (torch.device(f"cuda:{GPU}")
              if GPU is not None and torch.cuda.is_available()
              else torch.device("cpu"))
    logger.info(f"Device: {device}")

    logger.info(f"Loading search info: {SEARCH_INFO}")
    search_results = torch.load(
        SEARCH_INFO, map_location="cpu", weights_only=False)

    logger.info(f"Building test DataLoader from {DATA_DIR}/test ...")
    test_loader, classes = build_test_loader(
        DATA_DIR, IMG_SIZE, BATCH_SIZE, WORKERS, device.type == "cuda")
    n_test = len(test_loader.dataset)
    logger.info(f"  {n_test} test images | classes: {classes}")
    logger.info(f"  Positive class (F1): '{classes[1]}' (index 1)")

    master_csv_path = os.path.join(OUT_DIR, "all_epochs_results.csv")
    master_csv_f    = open(master_csv_path, "w", newline="")
    master_writer   = csv.writer(master_csv_f)
    master_writer.writerow(
        ["ops_key", "epoch", "val_acc1_stored",
         "acc_top1_test", "f1", "precision", "recall",
         "tp", "fp", "fn", "tn"])

    summary = {}   # key -> (epoch, acc, f1, prec, rec)

    for key in KEYS:
        key_dir = Path(BASE_DIR) / f"finetuned_ops_key{key}"
        epochs  = sorted([
            (int(m.group(1)), str(f))
            for f in key_dir.glob("epoch_*.pth.tar")
            if (m := re.match(r"epoch_(\d+)\.pth\.tar", f.name))
        ])

        logger.info(f"\n{'='*65}")
        logger.info(f"Key {key}: {len(epochs)} epochs found — building static model ...")
        model = build_static_model(key, search_results, device, IMG_SIZE)

        key_csv_path = os.path.join(OUT_DIR, f"key{key}_epoch_results.csv")
        key_csv_f    = open(key_csv_path, "w", newline="")
        key_writer   = csv.writer(key_csv_f)
        key_writer.writerow(
            ["epoch", "val_acc1_stored",
             "acc_top1_test", "f1", "precision", "recall",
             "tp", "fp", "fn", "tn"])

        best      = (None, 0.0, 0.0, 0.0, 0.0)
        best_path = None

        for ep, ckpt_path in epochs:
            val_acc1_stored = load_epoch_weights(model, ckpt_path, device)
            acc, f1, prec, rec, tp, fp, fn, tn = evaluate(
                model, test_loader, device)

            star = " ***" if f1 > best[2] else ""
            logger.info(
                f"  Key {key} ep {ep:2d}: "
                f"stored={val_acc1_stored:.2f}%  "
                f"test_acc={acc:.2f}%  F1={f1:.4f}  "
                f"Prec={prec:.4f}  Rec={rec:.4f}"
                f"{star}"
            )

            row = [key, ep, f"{val_acc1_stored:.4f}",
                   f"{acc:.4f}", f"{f1:.6f}",
                   f"{prec:.6f}", f"{rec:.6f}",
                   tp, fp, fn, tn]
            master_writer.writerow(row)
            master_csv_f.flush()
            key_writer.writerow(row[1:])  # drop ops_key column
            key_csv_f.flush()

            if f1 > best[2]:
                best      = (ep, acc, f1, prec, rec)
                best_path = ckpt_path

        key_csv_f.close()

        ep, acc, f1, prec, rec = best
        dst_name = (f"nasbnn_key{key}_best_ep{ep}"
                    f"_acc{acc:.2f}_f1{f1:.4f}.pth.tar")
        shutil.copy2(best_path, os.path.join(BEST_DIR, dst_name))
        summary[key] = best
        logger.info(
            f"  => Key {key} BEST: epoch {ep}  "
            f"TestAcc={acc:.2f}%  F1={f1:.4f}  -> {dst_name}"
        )

    master_csv_f.close()

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*72}")
    logger.info(
        f"  KEY   BEST EP   TEST ACC    F1        PRECISION   RECALL")
    logger.info(f"{'-'*72}")
    for key, (ep, acc, f1, prec, rec) in summary.items():
        logger.info(
            f"  Key {key}    ep {ep:2d}    {acc:.2f}%     "
            f"{f1:.4f}     {prec:.4f}      {rec:.4f}"
        )
    logger.info(f"{'='*72}")
    logger.info(f"\nAll results:      {OUT_DIR}/")
    logger.info(f"Best checkpoints: {BEST_DIR}/")
    logger.info(f"Master CSV:       {master_csv_path}")
    logger.info(f"Summary log:      {log_path}")


if __name__ == "__main__":
    main()
