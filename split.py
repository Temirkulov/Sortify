import os
import shutil
import random
from pathlib import Path
from collections import Counter

# ==========================
# Configuration
# ==========================
DATASET_DIR = Path("trashnet")      # Original TrashNet folder
OUTPUT_DIR = Path("trashnet-split")                   # Output directory

TRAIN_RATIO = 0.70
VAL_RATIO = 0.13
TEST_RATIO = 0.17

RANDOM_SEED = 42

# ==========================
# Setup
# ==========================
random.seed(RANDOM_SEED)

classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])

print("=" * 60)
print("Preparing TrashNet Dataset")
print("=" * 60)

print(f"Classes: {classes}")

# Remove old output directory if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# Statistics
summary = {}

# ==========================
# Process each class
# ==========================
for cls in classes:

    images = list((DATASET_DIR / cls).glob("*"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    summary[cls] = {
        "total": total,
        "train": len(train_imgs),
        "val": len(val_imgs),
        "test": len(test_imgs),
    }

    # Copy files
    for img in train_imgs:
        shutil.copy2(img, OUTPUT_DIR / "train" / cls / img.name)

    for img in val_imgs:
        shutil.copy2(img, OUTPUT_DIR / "val" / cls / img.name)

    for img in test_imgs:
        shutil.copy2(img, OUTPUT_DIR / "test" / cls / img.name)

# ==========================
# Summary
# ==========================
print("\nDataset Split Summary")
print("-" * 60)

total_train = total_val = total_test = total_images = 0

for cls, stats in summary.items():
    print(
        f"{cls:<12}"
        f"Total: {stats['total']:>3} | "
        f"Train: {stats['train']:>3} | "
        f"Val: {stats['val']:>3} | "
        f"Test: {stats['test']:>3}"
    )

    total_images += stats["total"]
    total_train += stats["train"]
    total_val += stats["val"]
    total_test += stats["test"]

print("-" * 60)
print(f"Total Images : {total_images}")
print(f"Train        : {total_train}")
print(f"Validation   : {total_val}")
print(f"Test         : {total_test}")

print("\nDataset successfully prepared!")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
