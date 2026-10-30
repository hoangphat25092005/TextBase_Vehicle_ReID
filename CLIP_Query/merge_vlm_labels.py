"""Labels for a fair CLIP-vs-VLM comparison: VLM captions for the train split, hand-checked ones for val/test.

The VLM file also relabels val/test, but those captions are what training is scored against (best-epoch
choice and text->image retrieval), so they stay the hand-checked ones from data/text_labels.json. Only the
captions the model trains on differ between the two runs.

    python -m CLIP_Query.merge_vlm_labels
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Merge VLM train labels with the hand-checked val/test labels")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--base", default=None, help="Defaults to <data_dir>/text_labels.json")
    parser.add_argument("--vlm", default=None, help="Defaults to <data_dir>/text_labels_vlm.json")
    parser.add_argument("--out", default=None, help="Defaults to <data_dir>/text_labels_vlm_train.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(args.base or data_dir / "text_labels.json") as f:
        base = json.load(f)
    with open(args.vlm or data_dir / "text_labels_vlm.json") as f:
        vlm = json.load(f)
    assert "train" in vlm["meta"]["splits"], "the VLM file has no train split: run generate_vlm_labels with train"

    vehicles, replaced = {}, 0
    for key, entry in base["vehicles"].items():
        if entry["split"] == "train":
            entry = vlm["vehicles"][key]   # KeyError = a train vehicle the VLM never described
            replaced += 1
        vehicles[key] = entry
    out = Path(args.out) if args.out else data_dir / "text_labels_vlm_train.json"
    with open(out, "w") as f:
        json.dump({"meta": {**base["meta"], "labeler_train": vlm["meta"]["labeler"]}, "vehicles": vehicles}, f, indent=2)
    flagged = sum(v["needs_review"] for v in vehicles.values() if v["split"] == "train")
    print(f"Wrote {out}: {replaced} train vehicles from the VLM ({flagged} flagged), rest from {base['meta']['model']} labels")


if __name__ == "__main__":
    main()
