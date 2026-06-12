import argparse
import random
import shutil
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets

from app.config import CLASSES, IMG_EXTS


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42



def collect_images(directory: Path) -> list[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(directory.glob(ext))
    return files


def copy_files(files: list[Path], dst_dir: Path, prefix: str = "") -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing_stems = {f.stem for f in dst_dir.iterdir()} if dst_dir.exists() else set()
    copied = 0
    skipped = 0

    for f in files:
        new_name = f"{prefix}{f.name}" if prefix else f.name
        new_stem = Path(new_name).stem

        if new_stem in existing_stems:
            skipped += 1
            continue

        shutil.copy(f, dst_dir / new_name)
        existing_stems.add(new_stem)
        copied += 1

    if skipped:
        print(f"  Skipped {skipped:,} duplicates")

    return copied


def split_rsna(rsna_src: Path, dst: Path):
    print("=" * 55)
    print("  STEP 1 - Splitting RSNA dataset (70 / 15 / 15)")
    print("=" * 55)
    random.seed(SEED)

    for cls in CLASSES:
        src_cls = rsna_src / cls
        if not src_cls.exists():
            raise FileNotFoundError(
                f"Expected '{src_cls}'.\n"
                f"Make sure rsna_raw/ has NORMAL/ and PNEUMONIA/ subdirectories"
            )

        files = collect_images(src_cls)
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val": files[n_train + n_val :],
        }

        print(f"\n  {cls}  (total: {n_total:,})")
        for split_name, split_files in splits.items():
            n = copy_files(split_files, dst / split_name / cls, prefix="rsna_")
            print(f"  {split_name:6s}: {n:>6,}")


def merge_chestxray(cxr_src: Path, dst: Path):
    print("\n" + "=" * 55)
    print("  STEP 2 - Merging ChestXRay2017 into train/")
    print("  (val/ and test/ untouched - RSNA-only for clean eval)")
    print("=" * 55)

    source_splits = ["train", "test"]

    for cls in CLASSES:
        total_copied = 0
        print(f"\n {cls}")
        for split in source_splits:
            src_cls = cxr_src / split / cls
            if not src_cls.exists():
                print(f"  {split:6s}: not found, skipping")
                continue

            files = collect_images(src_cls)
            n = copy_files(files, dst / "train" / cls, prefix=f"cxr_{split}_")
            print(f"  {split:6s}: {n:>6,} added to train/")
            total_copied += n

        print(f"  total : {total_copied:>6,} images merged")


def print_summary(dst: Path):
    print("\n" + "=" * 55)
    print("  FINAL DATASET SUMMARY")
    print("=" * 55)

    grand_total = 0
    for split in ["train", "val", "test"]:
        totals = {}
        for cls in CLASSES:
            cls_path = dst / split / cls
            totals[cls] = len(collect_images(cls_path)) if cls_path.exists() else 0

        total = sum(totals.values())
        grand_total += total
        ratio = totals["NORMAL"] / max(totals["PNEUMONIA"], 1)

        print(f"\n {split.upper()}")
        for cls in CLASSES:
            print(f"  {cls:<12}: {totals[cls]:>7,}")
        print(f"  {'Total':<12}: {total:>7,} (ratio {ratio:.2f}:1 N:P)")

    print(f"\n  GRAND TOTAL: {grand_total:,} images")
    print(f"\n  Output directory: {dst.resolve()}")
    print("\n  Next steps:")
    print("    1. Verify ratio above — update pos_weight in train.py if needed")
    print("    2. uv run python src/train.py")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Butyag training dataset from RSNA + optional ChestXRay2017"
    )
    parser.add_argument("--rsna", required=True,
                        help="Path to RSNA raw data (contains NORMAL/ and PNEUMONIA/)")
    parser.add_argument("--cxr", default=None,
                        help="Path to ChestXRay2017 (optional, contains train/ test/ val/)")
    parser.add_argument("--dst", default="data",
                        help="Output directory (will be created, default: data/)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete dst/ before starting (fresh build)")
    args = parser.parse_args()

    dst = Path(args.dst)

    if args.clean and dst.exists():
        print(f"--clean flag set. Removing existing {dst}/ ...")
        shutil.rmtree(dst)

    split_rsna(Path(args.rsna), dst)

    if args.cxr:
        merge_chestxray(Path(args.cxr), dst)
    else:
        print("\n  --cxr not provided. Skipping ChestXRay2017 merge.")

    print_summary(dst)


if __name__ == '__main__':
    main()
