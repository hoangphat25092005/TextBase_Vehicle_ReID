"""Measure text -> vehicle retrieval quality on the labelled test crops.

1. Create a label template (vehicle type is filled from the filename):
       python -m CLIP_Query.evaluate --init
2. Fill in the `color` column of CLIP_Query/eval_labels.csv (e.g. white, black, red, silver, blue).
3. Run the evaluation (no Chroma server needed):
       python -m CLIP_Query.evaluate
       python -m CLIP_Query.evaluate --model ViT-B-16 --pretrained laion2b_s34b_b88k
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from CLIP_Query.clip_encoder import ClipEncoder

ROOT = Path(__file__).resolve().parents[1]


def init_labels(image_dir: Path, labels_path: Path):
    if labels_path.exists():
        print(f"[Eval] {labels_path} already exists, not overwriting")
        return
    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        vehicle_type = image_path.stem.split("_")[0]
        rows.append({"image": image_path.name, "color": "", "type": vehicle_type})
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "color", "type"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Eval] Wrote {len(rows)} rows to {labels_path}. Fill in the color column, then rerun without --init.")


def average_precision(relevant: np.ndarray) -> float:
    """relevant: boolean array in ranked order."""
    if relevant.sum() == 0:
        return 0.0
    hits = np.cumsum(relevant)
    precisions = hits / np.arange(1, len(relevant) + 1)
    return float((precisions * relevant).sum() / relevant.sum())


def evaluate(encoder: ClipEncoder, image_dir: Path, labels_path: Path, ks=(1, 5, 10)):
    with open(labels_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["color"].strip()]
    if not rows:
        print("[Eval] No labelled rows (color column is empty).")
        return

    images = [cv2.imread(str(image_dir / r["image"])) for r in rows]
    keep = [i for i, img in enumerate(images) if img is not None]
    rows = [rows[i] for i in keep]
    image_feats = encoder.encode_images([images[i] for i in keep])

    colors = np.array([r["color"].strip().lower() for r in rows])
    types = np.array([r["type"].strip().lower() for r in rows])

    queries = {}
    for color, vehicle_type in sorted(set(zip(colors, types))):
        queries[f"{color} {vehicle_type}"] = (colors == color) & (types == vehicle_type)
    for color in sorted(set(colors)):
        queries[f"{color} vehicle"] = colors == color

    metrics = defaultdict(list)
    print(f"\n{'query':<24}{'#rel':>5}{'AP':>8}" + "".join(f"{'R@' + str(k):>7}" for k in ks))
    for query, relevant_mask in queries.items():
        scores = image_feats @ encoder.encode_text(query)
        ranked_relevant = relevant_mask[np.argsort(-scores)]
        ap = average_precision(ranked_relevant)
        metrics["AP"].append(ap)
        line = f"{query:<24}{int(relevant_mask.sum()):>5}{ap:>8.3f}"
        for k in ks:
            hit = float(ranked_relevant[:k].any())
            metrics[f"R@{k}"].append(hit)
            line += f"{hit:>7.0f}"
        print(line)

    print("\n[Eval] Summary over", len(queries), "queries,", len(rows), "images")
    print(f"  mAP  = {np.mean(metrics['AP']):.4f}")
    for k in ks:
        print(f"  R@{k:<3}= {np.mean(metrics[f'R@{k}']):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP text -> vehicle retrieval")
    parser.add_argument("--image_dir", default=str(ROOT / "data" / "images" / "test"))
    parser.add_argument("--labels", default=str(ROOT / "CLIP_Query" / "eval_labels.csv"))
    parser.add_argument("--init", action="store_true", help="Create the label template and exit")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    image_dir, labels_path = Path(args.image_dir), Path(args.labels)
    if args.init:
        init_labels(image_dir, labels_path)
        return
    evaluate(ClipEncoder(args.model, args.pretrained), image_dir, labels_path)


if __name__ == "__main__":
    main()
