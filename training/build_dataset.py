import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from collections import Counter

from app.config import CLASSES, IMG_EXTS

TRAIN_PY = Path("../butyag/training/train.py")


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    total = 0
    for ext in IMG_EXTS:
        total += len(list(directory.glob(ext)))
    return total


def format_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"

    return str(n)


def hr(char="-", width=55):
    print(char * width)


def validate_sources(rsna_src: Path, cxr_src: Path | None) -> dict:
    print("\n  Validating source directories...")
    counts = {"rsna": {}, "cxr": {}}
    errors = []

    for cls in CLASSES:
        cls_path = rsna_src / cls
        if not cls_path.exists():
            errors.append(f"Missing: {cls_path}")
            counts["rsna"][cls] = 0
        else:
            n = count_images(cls_path)
            counts["rsna"][cls] = n
            status = "OK" if n > 0 else "EMPTY"
            print(f"  rsna/{cls:<12}: {format_size(n):>7} images  [{status}]")
            if n == 0:
                errors.append(f"Empty directory: {cls_path}")

    if cxr_src:
        for split in ["train", "test"]:
            for cls in CLASSES:
                cls_path = cxr_src / split / cls
                key = f"{split}/{cls}"
                if not cls_path.exists():
                    print(f"  cxr/{key:<18}: not found (skipped)")
                    counts["cxr"][key] = 0
                else:
                    n = count_images(cls_path)
                    counts["cxr"][key] = n
                    print(f"  cxr/{key:<18}: {format_size(n):>7} images")

    if errors:
        print("\n ERRORS:")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("  All sources valid")
    return counts


def check_disk_space(counts: dict, dst: Path):
    KB_PER_IMAGE = 100
    rsna_total = sum(counts["rsna"].values())
    cxr_total = sum(counts["cxr"].values())
    total_imgs = rsna_total + cxr_total
    est_mb = (total_imgs * KB_PER_IMAGE) / 1024

    free_mb = shutil.disk_usage(dst.parent if dst.parent.exists() else Path(".")).free / 1024 / 1024

    print(f"\n Estimated output size: ~{est_mb:,.0f} MB")
    print(f"  Available disk space: {free_mb:,.0f} MB")

    if est_mb > free_mb * 0.9:
        print("\n WARNING: May not have enough disk space. Consider using --dst on a larger drive.")
    else:
        print("  Disk space")


def run_prepare(rsna_src: Path, cxr_src: Path | None, dst: Path, clean: bool):
    hr()
    print("  Running dataset preparation...")
    hr()

    cmd = [
        sys.executable, "-m", "training.prepare_dataset_rsna",
        "--rsna", str(rsna_src),
        "--dst", str(dst)
    ]
    if cxr_src:
        cmd += ["--cxr", str(cxr_src)]
    if clean:
        cmd += ["--clean"]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n prepare_dataset_rsna.py failed. Exiting.")
        sys.exit(1)


def verify_output(dst: Path) -> dict:
    print("\n Verifying output structure...")
    final_counts = {}
    errors = []

    for split in ["train", "val", "test"]:
        final_counts[split] = {}
        for cls in CLASSES:
            cls_path = dst / split / cls
            if not cls_path.exists():
                errors.append(f"Missing output: {cls_path}")
                final_counts[split][cls] = 0
            else:
                n = count_images(cls_path)
                final_counts[split][cls] = n
                if n == 0:
                    errors.append(f"Empty output: {cls_path}")

    if errors:
        print("  ERRORS after preparation:")
        for e in errors:
            print(f"    x {e}")
        sys.exit(1)

    print(" Output structure verified.")
    return final_counts


def update_pos_weight(final_counts: dict):
    train_normal = final_counts["train"]["NORMAL"]
    train_pneumonia = final_counts["train"]["PNEUMONIA"]

    if train_pneumonia == 0:
        print("\n WARNING: No pneumonia images found - skipping pos_weight update")
        return

    ratio = round(train_normal / train_pneumonia, 2)
    print(f"\n Train Normal:Pneumonia ratio: {ratio:.2f}:1")
    print(f" Recommended pos_weight: {ratio}")

    if not TRAIN_PY.exists():
        print(f" WARNING: {TRAIN_PY} not found - update pos_weight manually")
        print(f" Set CONFIG['pos_weight'] = {ratio} in train.py")
        return

    content = TRAIN_PY.read_text()
    pattern = r'("pos_weight"\s*:s*)[\d.]+'
    new_content = re.sub(pattern, rf'\g<1>{ratio}', content)

    if new_content == content:
        print(f" WARNING: Could not find pos_weight in {TRAIN_PY} - update manually")
    else:
        TRAIN_PY.write_text(new_content)
        print(f" Auto-updated pos_weight = {ratio} in {TRAIN_PY}")


def print_report(final_counts: dict, dst: Path):
    hr("=")
    print(" DATASET READY - FINAL REPORT")
    hr("=")

    for split in ["train", "val", "test"]:
        normal = final_counts[split]["NORMAL"]
        pneumonia = final_counts[split]["PNEUMONIA"]
        total = normal + pneumonia
        ratio = normal / max(pneumonia, 1)

        print(f"\n  {split.upper():<6}  {total:>7,} images")
        print(f"    NORMAL    : {normal:>7,}  ({normal / total * 100:.1f}%)")
        print(f"    PNEUMONIA : {pneumonia:>7,}  ({pneumonia / total * 100:.1f}%)")
        print(f"    Ratio     : {ratio:.2f}:1  (Normal:Pneumonia)")

    hr()
    print(f"\n  Output  : {dst.resolve()}")
    print(f"  Ready to train. Run:")
    print(f"\n    cd ../butyag/src && uv run python train.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Butyag end-to-end dataset builder"
    )
    parser.add_argument("--rsna", required=True,
                        help="Path to RSNA raw data (NORMAL/ and PNEUMONIA/ subdirs)")
    parser.add_argument("--cxr", default=None,
                        help="Path to ChestXRay2017 (optional)")
    parser.add_argument("--dst", default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe dst/ before building")
    args = parser.parse_args()

    rsna_src = Path(args.rsna)
    cxr_src = Path(args.cxr) if args.cxr else None
    dst = Path(args.dst)

    hr("=")
    print("  BUTYAG — DATASET BUILD AUTOMATION")
    hr("=")
    print(f"\n  RSNA source      : {rsna_src}")
    print(f"  CXR2017 source   : {cxr_src or 'not provided'}")
    print(f"  Output directory : {dst}")
    print(f"  Clean build      : {args.clean}")

    # 1. Validate
    hr()
    print("  STEP 1 — Validate sources")
    hr()
    counts = validate_sources(rsna_src, cxr_src)

    # 2. Disk space
    hr()
    print("  STEP 2 — Disk space check")
    hr()
    check_disk_space(counts, dst)

    # 3. Prepare
    run_prepare(rsna_src, cxr_src, dst, args.clean)

    # 4. Verify
    hr()
    print("  STEP 4 — Verify output")
    hr()
    final_counts = verify_output(dst)

    # 5. Update pos_weight
    hr()
    print("  STEP 5 — Sync pos_weight in train.py")
    hr()
    update_pos_weight(final_counts)

    # 6. Report
    print_report(final_counts, dst)


if __name__ == '__main__':
    main()