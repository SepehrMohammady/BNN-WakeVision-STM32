"""
prepare_full_wakevision.py
--------------------------
Extracts the full WakeVision dataset from tar.gz archives, resizes images to
TARGET_SIZE, and organises them into:

  data/WakeVision_Full/
    train/
      person_present/
      no_person_present/
    val/
      person_present/
      no_person_present/

The script processes archives sequentially but resizes images in parallel
using a thread-pool, keeping memory and temp-disk usage minimal.

Usage:
  python prepare_full_wakevision.py [--workers N] [--quality Q] [--force]
"""

import argparse
import csv
import io
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR    = Path("./data/WakeVision_From_Local_SSD_V3")
OUTPUT_DIR  = Path("./data/WakeVision_Full")
TARGET_SIZE = (128, 128)   # Must match model input size
JPEG_QUALITY = 90          # JPEG quality for saved images (0-95)
SEED        = 42

TRAIN_CSV   = DATA_DIR / "wake_vision_train_large.csv"
VAL_CSV     = DATA_DIR / "wake_vision_validation.csv"
TEST_CSV    = DATA_DIR / "wake_vision_test.csv"
LABEL_COL   = "person"
FILE_COL    = "filename"
CLASS_MAP   = {1: "person_present", 0: "no_person_present"}
# ───────────────────────────────────────────────────────────────────────────────


def load_label_map(csv_path: Path) -> dict:
    """Return {filename: label_int} from a CSV with 'filename' and 'person' cols.
    Only entries with labels present in CLASS_MAP (0 or 1) are kept;
    ambiguous samples (-1) are silently discarded."""
    label_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row[FILE_COL].strip()
            try:
                label = int(float(row[LABEL_COL]))
            except (ValueError, KeyError):
                continue
            if label not in CLASS_MAP:  # skip -1 (ambiguous) and any other unknowns
                continue
            label_map[fname] = label
    return label_map


def resize_and_save(image_bytes: bytes, dest_path: Path, size: tuple, quality: int):
    """Decode bytes → resize → save as JPEG. Returns True on success."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img.save(dest_path, "JPEG", quality=quality)
        return True
    except Exception:
        return False


def process_archive(
    archive_path: Path,
    label_map: dict,
    output_dir: Path,
    size: tuple,
    quality: int,
    num_workers: int,
) -> tuple:
    """
    Stream one tar.gz archive, submit resize+save tasks to a thread pool.
    Returns (processed, skipped_no_label, skipped_error).
    """
    processed = skipped_no_label = skipped_error = 0
    futures = {}

    with tarfile.open(archive_path, "r:gz") as tf, \
         ThreadPoolExecutor(max_workers=num_workers) as executor:

        for member in tf:
            if not member.isfile():
                continue
            fname = os.path.basename(member.name)
            label = label_map.get(fname)
            if label is None:
                skipped_no_label += 1
                continue

            class_dir = output_dir / CLASS_MAP[label]
            dest = class_dir / fname
            if dest.exists():
                processed += 1
                continue

            fobj = tf.extractfile(member)
            if fobj is None:
                skipped_error += 1
                continue
            image_bytes = fobj.read()

            future = executor.submit(resize_and_save, image_bytes, dest, size, quality)
            futures[future] = fname

        for future in as_completed(futures):
            if future.result():
                processed += 1
            else:
                skipped_error += 1

    return processed, skipped_no_label, skipped_error


def prepare_split(
    archives: list,
    label_map: dict,
    split_dir: Path,
    size: tuple,
    quality: int,
    num_workers: int,
    split_name: str,
):
    # Create class directories
    for cls in CLASS_MAP.values():
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

    total_proc = total_skip_lbl = total_skip_err = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {len(archives)} archive(s)")
    print(f"Output → {split_dir}")
    print(f"Labels in CSV: {len(label_map):,}")
    print(f"{'='*60}")

    for archive_path in tqdm(archives, desc=split_name, unit="archive"):
        p, sl, se = process_archive(
            archive_path, label_map, split_dir, size, quality, num_workers
        )
        total_proc     += p
        total_skip_lbl += sl
        total_skip_err += se
        tqdm.write(
            f"  {archive_path.name}: +{p} saved | "
            f"no_label={sl} | errors={se}"
        )

    elapsed = time.time() - t0
    print(f"\n{split_name} complete in {elapsed/60:.1f} min")
    print(f"  Saved:      {total_proc:,}")
    print(f"  No label:   {total_skip_lbl:,}")
    print(f"  Errors:     {total_skip_err:,}")

    # Count actual files
    for cls in CLASS_MAP.values():
        n = len(list((split_dir / cls).glob("*.jpg")))
        print(f"  {split_dir / cls} → {n:,} images")


def main():
    parser = argparse.ArgumentParser(description="Prepare full WakeVision dataset")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Threads per archive for resize+save (default: 8)"
    )
    parser.add_argument(
        "--quality", type=int, default=JPEG_QUALITY,
        help=f"JPEG output quality (default: {JPEG_QUALITY})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing output directory and start fresh"
    )
    parser.add_argument(
        "--val-only", action="store_true",
        help="Only prepare the validation split (for quick testing)"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only prepare the test split (skip train and val)"
    )
    parser.add_argument(
        "--include-test", action="store_true",
        help="Also prepare the test split (in addition to val + train)"
    )
    args = parser.parse_args()

    if args.force and OUTPUT_DIR.exists():
        import shutil
        print(f"--force: removing {OUTPUT_DIR} ...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Test-only shortcut ────────────────────────────────────────────────────
    if args.test_only:
        _prepare_test_split(args)
        print("\nDone. Test split prepared at:", (OUTPUT_DIR / "test").resolve())
        return

    # ── Validation split ──────────────────────────────────────────────────────
    val_label_map = load_label_map(VAL_CSV)
    print(f"Val CSV: {len(val_label_map):,} labelled images")

    val_archives = sorted(DATA_DIR.glob("wake-vision-validation.tar.gz"))
    if not val_archives:
        print("ERROR: wake-vision-validation.tar.gz not found in", DATA_DIR)
        return

    prepare_split(
        archives=val_archives,
        label_map=val_label_map,
        split_dir=OUTPUT_DIR / "val",
        size=TARGET_SIZE,
        quality=args.quality,
        num_workers=args.workers,
        split_name="Validation",
    )

    if args.val_only:
        print("\n--val-only: skipping training and test splits.")
        return

    # ── Training split ────────────────────────────────────────────────────────
    train_label_map = load_label_map(TRAIN_CSV)
    print(f"\nTrain CSV: {len(train_label_map):,} labelled images")

    # Sort numerically: wake-vision-train-10.tar.gz ... wake-vision-train-99.tar.gz
    train_archives = sorted(
        DATA_DIR.glob("wake-vision-train-*.tar.gz"),
        key=lambda p: int(p.name.split(".")[0].rsplit("-", 1)[-1])
    )
    if not train_archives:
        print("ERROR: No wake-vision-train-*.tar.gz found in", DATA_DIR)
        return

    print(f"Found {len(train_archives)} training archives")
    prepare_split(
        archives=train_archives,
        label_map=train_label_map,
        split_dir=OUTPUT_DIR / "train",
        size=TARGET_SIZE,
        quality=args.quality,
        num_workers=args.workers,
        split_name="Train",
    )

    # ── Test split (opt-in) ───────────────────────────────────────────────────
    if args.include_test:
        _prepare_test_split(args)

    print("\nDone. Dataset prepared at:", OUTPUT_DIR.resolve())


def _prepare_test_split(args):
    """Extract and resize the test split."""
    if not TEST_CSV.exists():
        print(f"ERROR: Test CSV not found: {TEST_CSV}")
        return

    test_label_map = load_label_map(TEST_CSV)
    print(f"\nTest CSV: {len(test_label_map):,} labelled images")

    test_archives = sorted(DATA_DIR.glob("wake-vision-test.tar.gz"))
    if not test_archives:
        print("ERROR: wake-vision-test.tar.gz not found in", DATA_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_split(
        archives=test_archives,
        label_map=test_label_map,
        split_dir=OUTPUT_DIR / "test",
        size=TARGET_SIZE,
        quality=args.quality,
        num_workers=args.workers,
        split_name="Test",
    )


if __name__ == "__main__":
    main()
