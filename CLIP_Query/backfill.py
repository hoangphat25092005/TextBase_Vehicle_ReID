"""Index crops already stored in the ReID ChromaDB collections into the CLIP collection.

Usage (Chroma server must be running, e.g. via main_stream.py or `chroma run`):
    python -m CLIP_Query.backfill
    python -m CLIP_Query.backfill --source_prefix vehicle_embeddings --host localhost --port 8000
"""
import argparse

from CLIP_Query.clip_encoder import ClipEncoder
from CLIP_Query.clip_index import ClipIndex, VEHICLE_TYPES, decode_image, make_client, make_doc_id


def backfill(index: ClipIndex, client, source_prefix: str, page_size: int = 256) -> int:
    total_added = 0
    for type_id, type_name in VEHICLE_TYPES.items():
        name = f"{source_prefix}_{type_name}"
        try:
            source = client.get_collection(name=name)
        except Exception:
            print(f"[Backfill] Skipping missing collection: {name}")
            continue

        count = source.count()
        print(f"[Backfill] {name}: {count} entries")
        for offset in range(0, count, page_size):
            page = source.get(limit=page_size, offset=offset, include=["metadatas"])
            items = []
            for metadata in page['metadatas']:
                encoded = metadata.get('image_encoded')
                if not encoded or metadata.get('vehicle_id') is None:
                    continue
                image = decode_image(encoded)
                if image is None:
                    continue
                camera_id = metadata.get('camera_id', 'unknown')
                items.append({
                    "id": make_doc_id(type_id, metadata['vehicle_id'], camera_id),
                    "image": image,
                    "metadata": metadata,
                })
            added = index.add_many(items)
            total_added += added
            print(f"[Backfill] offset {offset}: added {added}/{len(items)}")
    return total_added


def main():
    parser = argparse.ArgumentParser(description="Backfill CLIP embeddings from ReID collections")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source_prefix", default="vehicle_embeddings_streaming",
                        help="ReID collection prefix (vehicle_embeddings_streaming or vehicle_embeddings)")
    parser.add_argument("--collection", default="vehicle_clip")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    client = make_client(args.host, args.port)
    index = ClipIndex(client, ClipEncoder(args.model, args.pretrained), collection_name=args.collection)
    added = backfill(index, client, args.source_prefix)
    print(f"[Backfill] Done. Added {added} entries, collection now has {index.count()}")


if __name__ == "__main__":
    main()
