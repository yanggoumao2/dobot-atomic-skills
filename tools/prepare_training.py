"""Add training-only normalization to an exported ACT dataset."""
import argparse
import json
from pathlib import Path
from scripts.act_training_adapter import training_stats

def prepare(directory):
    directory = Path(directory)
    split = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    if not split["train"] or not split["validation"]:
        raise ValueError("Need nonempty train and validation episodes (collect at least two successes)")
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Train/validation overlap")
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["camera_names"] != ["top", "left"]:
        raise ValueError("Current ACT host trainer expects top,left cameras")
    stats = training_stats(directory, split["train"])
    target = directory / "train_normalization_stats.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump({key: value.tolist() for key, value in stats.items()}, f, indent=2)
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    print(prepare(parser.parse_args().dataset))
