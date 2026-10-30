"""Text search over indexed vehicles from the command line.

Usage:
    python -m CLIP_Query.search_cli "white truck" --top_k 10
    python -m CLIP_Query.search_cli "red car in cam2" --save results/clip_search
"""
import argparse
from pathlib import Path

import cv2

from CLIP_Query.clip_encoder import ClipEncoder
from CLIP_Query.clip_index import ClipIndex, decode_image, make_client


def main():
    parser = argparse.ArgumentParser(description="CLIP text -> vehicle search")
    parser.add_argument("query")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--collection", default="vehicle_clip")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--save", default=None, help="Directory to save result crops")
    args = parser.parse_args()

    index = ClipIndex(make_client(args.host, args.port),
                      ClipEncoder(args.model, args.pretrained),
                      collection_name=args.collection)
    print(f"[Search] {index.count()} indexed crops")

    results = index.search(args.query, top_k=args.top_k)
    if not results:
        print("No results.")
        return

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for rank, vehicle in enumerate(results, start=1):
        cameras = ", ".join(a["camera_id"] for a in vehicle["appearances"])
        print(f"{rank:>2}. {vehicle['vehicle_type_name']} ID {vehicle['vehicle_id']:<4} "
              f"score={vehicle['score']:.3f}  cameras=[{cameras}]")
        if save_dir:
            for appearance in vehicle["appearances"]:
                if not appearance.get("image_encoded"):
                    continue
                image = decode_image(appearance["image_encoded"])
                name = f"{rank:02d}_{vehicle['vehicle_type_name']}_{vehicle['vehicle_id']}_{appearance['camera_id']}.jpg"
                cv2.imwrite(str(save_dir / name), image)

    if save_dir:
        print(f"[Search] Crops saved to {save_dir}")


if __name__ == "__main__":
    main()
