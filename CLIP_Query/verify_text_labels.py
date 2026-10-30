"""Sanity-check data/text_labels.json and write an HTML page for manual review.

    python -m CLIP_Query.verify_text_labels
    python -m CLIP_Query.verify_text_labels --cross_model ViT-B-16 --cross_pretrained openai

Hard checks (non-zero exit on failure): every split image is labelled exactly once, captions
are non-empty and consistent with their attributes. Everything else is reported for judgement.
Note that the retrieval check uses the same CLIP that produced the labels, so it measures
self-consistency; --cross_model relabels with a second CLIP for an independent agreement score.
"""
import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from CLIP_Query.evaluate import average_precision
from CLIP_Query.generate_text_labels import (ATTRIBUTES, SPLIT_DIRS, generate, make_caption,
                                             parse_name, query_phrases, read_split)

ROOT = Path(__file__).resolve().parents[1]


def check_coverage(vehicles, data_dir, min_conf):
    errors = []
    labelled = Counter(img for v in vehicles.values() for img in v["images"])
    for img, count in labelled.items():
        if count > 1:
            errors.append(f"{img} appears under {count} vehicles")
    for split in SPLIT_DIRS:
        for img_name in read_split(data_dir, split):
            key, _, _ = parse_name(img_name)
            if key not in vehicles or img_name not in vehicles[key]["images"]:
                errors.append(f"{split}: {img_name} has no label")
    for key, v in vehicles.items():
        spec = ATTRIBUTES.get(v["type"])
        if spec is None:
            errors.append(f"{key}: unknown type {v['type']!r}")
            continue
        if not v["caption"].strip():
            errors.append(f"{key}: empty caption")
        invalid = [name for name, options in spec.items() if v["attributes"].get(name) not in options]
        for name in invalid:
            errors.append(f"{key}: {name}={v['attributes'].get(name)!r} is not one of {list(spec[name])}")
        if not invalid and v["caption"] != make_caption(v["type"], v["attributes"], v["confidence"], min_conf,
                                                        v.get("keep")):
            errors.append(f"{key}: caption {v['caption']!r} does not match its attributes")
    return errors


def report_stats(vehicles):
    images = [(v["type"], info) for v in vehicles.values() for info in v["images"].values()]
    type_acc = np.mean([info["clip_type"] == t for t, info in images])
    print(f"\n[Type sanity] CLIP zero-shot type == filename type: {type_acc:.3f} over {len(images)} images"
          + ("" if type_acc >= 0.8 else "  <-- LOW: CLIP may be unreliable on these crops"))
    for vehicle_type in ATTRIBUTES:
        preds = Counter(info["clip_type"] for t, info in images if t == vehicle_type)
        if preds:
            print(f"  {vehicle_type:<11} n={sum(preds.values()):<4} predicted as {dict(preds)}")

    print("\n[Per type] attribute distribution | cross-camera agreement | mean confidence")
    for vehicle_type, spec in ATTRIBUTES.items():
        subset = [v for v in vehicles.values() if v["type"] == vehicle_type]
        if not subset:
            continue
        print(f"  {vehicle_type} ({len(subset)} vehicles)")
        for name in spec:
            counts = Counter(v["attributes"][name] for v in subset).most_common()
            agree = np.mean([len({i[name] for i in v["images"].values()}) == 1 for v in subset])
            conf = np.mean([v["confidence"][name] for v in subset])
            print(f"    {name:<12} agree {agree:.2f}  conf {conf:.2f}  {dict(counts)}")

    captions = Counter(v["caption"] for v in vehicles.values())
    print(f"\n  distinct captions: {len(captions)} for {len(vehicles)} vehicles; most common:")
    for caption, n in captions.most_common(3):
        print(f"    {n:>3} x {caption}")
    flagged = sum(v["needs_review"] for v in vehicles.values())
    print(f"  needs_review: {flagged} ({flagged / len(vehicles):.1%})")


def report_retrieval(vehicles, data_dir, encoder, ks=(1, 5)):
    """Phrase -> test image retrieval with plain CLIP (self-consistency of the labels)."""
    test_images = read_split(data_dir, "test")
    keys = [parse_name(img)[0] for img in test_images]
    feats = encoder.encode_images([cv2.imread(str(data_dir / "images" / "test" / img)) for img in test_images])

    def run(name, phrase_of):
        values = np.array([phrase_of(vehicles[k]) or "" for k in keys])
        aps, hits = [], {k: [] for k in ks}
        for phrase in sorted(set(values) - {""}):
            relevant = (values == phrase)[np.argsort(-(feats @ encoder.encode_text(phrase)))]
            aps.append(average_precision(relevant))
            for k in ks:
                hits[k].append(float(relevant[:k].any()))
        line = f"  {name:<14} {len(aps):>3} queries  mAP {np.mean(aps):.3f}"
        line += "".join(f"  R@{k} {np.mean(hits[k]):.3f}" for k in ks)
        print(line)

    print(f"\n[Retrieval sanity] plain CLIP on {len(test_images)} test images")
    for kind in ("caption", "color", "style"):
        run(kind, lambda v: query_phrases(v)[kind])


def report_cross_model(vehicles, data_dir, encoder):
    other = generate(encoder, data_dir, min_conf=0.0)
    print("\n[Cross-model] agreement with a second CLIP (all | non-flagged)")
    for vehicle_type, spec in ATTRIBUTES.items():
        keys = [k for k, v in vehicles.items() if v["type"] == vehicle_type]
        confident = [k for k in keys if not vehicles[k]["needs_review"]]
        parts = []
        for name in spec:
            agree = lambda ks: np.mean([vehicles[k]["attributes"][name] == other[k]["attributes"][name]
                                        for k in ks]) if ks else float("nan")
            parts.append(f"{name} {agree(keys):.2f} | {agree(confident):.2f}")
        print(f"  {vehicle_type:<11} " + ",  ".join(parts))


def write_review_html(vehicles, out_path: Path):
    order = sorted(vehicles.items(), key=lambda kv: (not kv[1]["needs_review"], min(kv[1]["confidence"].values())))
    cards = []
    for key, v in order:
        thumbs = "".join(
            f'<figure><img loading="lazy" src="images/{SPLIT_DIRS[info["split"]]}/{html.escape(img)}">'
            f'<figcaption>cam{info["cam"]} · {info["view"]}<br>'
            + " · ".join(f'{info[name]} {info[name + "_conf"]:.2f}' for name in v["attributes"])
            + '</figcaption></figure>'
            for img, info in sorted(v["images"].items()))
        flag = '<span class="flag">review</span>' if v["needs_review"] else ""
        conf = " · ".join(f"{name} {c:.2f}" for name, c in v["confidence"].items())
        cards.append(
            f'<div class="card"><h3>{html.escape(key)} {flag}</h3>'
            f'<p class="cap">{html.escape(v["caption"])}</p>'
            f'<p class="meta">{v["split"]} · {conf}</p>'
            f'<div class="thumbs">{thumbs}</div></div>')
    out_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Text Label Review</title>
<style>
:root{{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--flag:#b4441c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--card:#22221f;--fg:#ecece8;--muted:#9a9a93;--flag:#f08a5d}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border-radius:8px;padding:10px}} h3{{margin:0;font-size:14px}}
.cap{{font-size:15px;font-weight:600;margin:4px 0}} .meta,figcaption{{color:var(--muted);font-size:12px}}
.flag{{color:var(--flag);font-size:12px;margin-left:6px}} .thumbs{{display:flex;gap:6px}}
figure{{margin:0;flex:1}} img{{width:100%;height:140px;object-fit:contain;background:#0001}}
</style></head><body><h1>Text label review ({len(vehicles)} vehicles, flagged first)</h1>
<p>To fix a label, edit its attributes and caption in text_labels.json.</p>
<div class="grid">{''.join(cards)}</div></body></html>""", encoding="utf-8")
    print(f"\n[Review] Wrote {out_path} (open it in a browser)")


def main():
    parser = argparse.ArgumentParser(description="Verify generated text labels")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--labels", default=None, help="Defaults to <data_dir>/text_labels.json")
    parser.add_argument("--no_clip", action="store_true", help="Skip the retrieval check (no model load)")
    parser.add_argument("--cross_model", default=None, help="Second CLIP model for an independent check")
    parser.add_argument("--cross_pretrained", default="openai")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels_path = Path(args.labels) if args.labels else data_dir / "text_labels.json"
    with open(labels_path) as f:
        labels = json.load(f)
    vehicles = labels["vehicles"]

    errors = check_coverage(vehicles, data_dir, labels["meta"]["min_conf"])
    print(f"[Coverage] {len(vehicles)} vehicles, "
          f"{sum(len(v['images']) for v in vehicles.values())} images: "
          + ("OK" if not errors else f"{len(errors)} errors"))
    for err in errors[:20]:
        print("  -", err)
    if errors:
        sys.exit(1)

    report_stats(vehicles)
    if not args.no_clip:
        from CLIP_Query.clip_encoder import ClipEncoder
        report_retrieval(vehicles, data_dir, ClipEncoder(labels["meta"]["model"], labels["meta"]["pretrained"]))
    if args.cross_model:
        from CLIP_Query.clip_encoder import ClipEncoder
        report_cross_model(vehicles, data_dir, ClipEncoder(args.cross_model, args.cross_pretrained))
    write_review_html(vehicles, data_dir / "text_labels_review.html")


if __name__ == "__main__":
    main()
