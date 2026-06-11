import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["NORMAL", "PNEUMONIA"]
VAL_RATIO = 0.15
SEED = 42


def prepare(src: Path, dst: Path):
    random.seed(SEED)

    print(f"Source: {src}")
    print(f"Dest: {dst}")
    print()

    print("Copying test set...")
    for cls in CLASSES:
        src_cls = src / "test" / cls
        dst_cls = dst / "test" / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        files = list(src_cls.glob("*.jpeg")) + list(src_cls.glob("*.jpg"))
        for f in files:
            shutil.copy(f, dst_cls / f.name)
        print(f"  test/{cls}: {len(files)} images")

    print()

    print(f"Splitting train → train ({int((1-VAL_RATIO)*100)}%) + val ({int(VAL_RATIO*100)}%)...")
    for cls in CLASSES:
        src_cls = src / "train" / cls
        files = list(src_cls.glob("*.jpeg")) + list(src_cls.glob("*.jpg"))
        random.shuffle(files)

        n_val = int(len(files) * VAL_RATIO)
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split, split_files in [("train", train_files), ("val", val_files)]:
            dst_cls = dst / split / cls
            dst_cls.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_cls / f.name)

        print(f"  train/{cls}: {len(train_files)} | val/{cls}: {len(val_files)}")

    print()

    print("Final split summary:")
    total = 0
    for split in ["train", "val", "test"]:
        split_total = 0
        for cls in CLASSES:
            n = len(list((dst / split / cls).glob("*")))
            split_total += n
        total += split_total
        print(f"  {split:6s}: {split_total:>5} images")
    print(f"  {'total':6s}: {total:>5} images")
    print(f"\nDone. Point dataset.py to: {dst}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="chest_xray",
                        help="Path to extracted Kaggle dataset (contains train/ test/ val/)")
    parser.add_argument("--dst", default="data", help="Output path (will be created)")
    args = parser.parse_args()

    prepare(Path(args.src), Path(args.dst))

