"""Text -> vehicle retrieval with a re-ID backbone trained with --text_labels, vs. plain CLIP.

    python -m CLIP_Query.evaluate_text_reid --model_path checkpoints/best_osnet_model.pth
    python -m CLIP_Query.evaluate_text_reid --model_path checkpoints/best_resnet_ibn_model.pth --split val

For every image in the split, three queries are built from data/text_labels.json (full caption,
color only, style only). Each distinct query is scored by mAP / R@1 / R@5 over all images of the
split; an image is relevant when its vehicle has that exact phrase. Scores are reported for all
queries and per vehicle type.
"""
import argparse
import json
from pathlib import Path

import cv2
import torch

from CLIP_Query.clip_encoder import ClipEncoder
from CLIP_Query.generate_text_labels import SPLIT_DIRS, parse_name, query_phrases, read_split
from modules.feature_extraction import VehicleDescriptor
from modules.text_alignment import text_retrieval_map

ROOT = Path(__file__).resolve().parents[1]

# feature_dim each model type is trained with in trainer.py
FEATURE_DIMS = {"osnet": 512, "resnet_ibn": 2048}


def main():
    parser = argparse.ArgumentParser(description="Evaluate text -> vehicle retrieval of a trained re-ID model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default=None, choices=list(FEATURE_DIMS),
                        help="Defaults to the model_type stored in the checkpoint")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--labels", default=None, help="Defaults to <data_dir>/text_labels.json")
    parser.add_argument("--split", default="test", choices=list(SPLIT_DIRS))
    parser.add_argument("--no_baseline", action="store_true", help="Skip the plain CLIP baseline")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(Path(args.labels) if args.labels else data_dir / "text_labels.json") as f:
        vehicles = json.load(f)["vehicles"]

    model_type = args.model_type or torch.load(args.model_path, map_location="cpu",
                                               weights_only=False).get("model_type", "osnet")
    descriptor = VehicleDescriptor(model_path=args.model_path, model_type=model_type,
                                   input_size=(224, 224), feature_dim=FEATURE_DIMS[model_type])
    if descriptor.text_head is None:
        raise SystemExit(f"{args.model_path} has no text head; train it with --text_labels")

    image_names = read_split(data_dir, args.split)
    images = [cv2.imread(str(data_dir / "images" / SPLIT_DIRS[args.split] / name)) for name in image_names]
    queries = [query_phrases(vehicles[parse_name(name)[0]]) for name in image_names]
    types = [parse_name(name)[1] for name in image_names]
    phrases = sorted({q[kind] for q in queries for kind in q} - {None})

    # The backbone was trained on RGB images (utils/dataset.py)
    model_embeds = torch.stack([descriptor.extract_text_aligned_feature(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                for img in images]).cpu()
    text_embeds = {p: descriptor.encode_text_query(p).cpu() for p in phrases}

    results = {"backbone + text head": model_embeds}
    if not args.no_baseline:
        # Same CLIP as the text encoder, so both rows use identical query embeddings
        clip = descriptor._clip_encoder
        results["plain CLIP (zero-shot)"] = torch.from_numpy(clip.encode_images(images))

    manual = sum(vehicles[parse_name(name)[0]].get("manual", False) for name in image_names)
    print(f"\n{args.split}: {len(images)} images ({manual} with hand-checked labels), "
          f"{len(phrases)} distinct phrases, model {model_type}")
    # Per type: only that type's queries, but every image of the split is ranked (other types are distractors)
    groups = [("all", None)] + [(t, t) for t in sorted(set(types))]
    for name, embeds in results.items():
        print(f"\n{name}")
        print(f"  {'query':<9}{'type':<12}{'#q':>4}{'mAP':>8}{'R@1':>8}{'R@5':>8}")
        for kind in ("caption", "color", "style"):
            for group, vehicle_type in groups:
                per_image = [q[kind] if vehicle_type in (None, t) else None for q, t in zip(queries, types)]
                n_queries = len(set(per_image) - {None})
                if not n_queries:
                    continue
                mAP, r1, r5 = text_retrieval_map(embeds, per_image, text_embeds)
                print(f"  {kind:<9}{group:<12}{n_queries:>4}{mAP:>8.3f}{r1:>8.3f}{r5:>8.3f}")
    if manual < len(image_names):
        print("\nNote: labels that are not hand-checked come from the same zero-shot CLIP, "
              "so the baseline is optimistic on them.")


if __name__ == "__main__":
    main()
