"""
One-time script to reorganize a raw CIFAKE (or similarly structured)
download into the data/train|val|test/real|fake layout this project expects.

Adjust SOURCE_TRAIN / SOURCE_TEST below to match wherever you unzipped the
dataset, then run:

    python3 prepare_data.py

It COPIES files (doesn't move), so your original download stays intact.
If your downloaded folder names differ (e.g. "REAL"/"FAKE" vs "real"/"fake"),
adjust the CLASS_MAP below accordingly.
"""

import random
import shutil
from pathlib import Path

# --- EDIT THESE THREE LINES to match your downloaded dataset ---
SOURCE_TRAIN = Path("~/Desktop/Project/cifake/train").expanduser()   # folder with REAL/ and FAKE/ subfolders
SOURCE_TEST = Path("~/Desktop/Project/cifake/test").expanduser()     # folder with REAL/ and FAKE/ subfolders
CLASS_MAP = {"REAL": "real", "FAKE": "fake"}                    # source_folder_name -> real/fake
# -----------------------------------------------------------------

DEST = Path("data")
VAL_FRACTION = 0.1  # carve 10% of train off for validation

random.seed(42)


def copy_files(file_list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fp in file_list:
        shutil.copy2(fp, dest_dir / fp.name)


def main():
    for src_name, dst_name in CLASS_MAP.items():
        src_dir = SOURCE_TRAIN / src_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Expected {src_dir} to exist -- check SOURCE_TRAIN and folder names")

        files = list(src_dir.iterdir())
        random.shuffle(files)
        n_val = int(len(files) * VAL_FRACTION)
        val_files = files[:n_val]
        train_files = files[n_val:]

        print(f"{dst_name}: {len(train_files)} train, {len(val_files)} val")
        copy_files(train_files, DEST / "train" / dst_name)
        copy_files(val_files, DEST / "val" / dst_name)

    for src_name, dst_name in CLASS_MAP.items():
        src_dir = SOURCE_TEST / src_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Expected {src_dir} to exist -- check SOURCE_TEST and folder names")
        files = list(src_dir.iterdir())
        print(f"{dst_name}: {len(files)} test")
        copy_files(files, DEST / "test" / dst_name)

    print("\nDone. Data organized under ./data/")


if __name__ == "__main__":
    main()