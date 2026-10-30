"""Generate per-vehicle text labels with zero-shot CLIP.

Every image is scored against fixed attribute lists (which depend on the vehicle type from the
filename); the probabilities are averaged over all images of the same vehicle (type, id) so both
cameras share one caption. Motorcycles are mostly seen from behind with the rider covering the
bike, so they are described by the rider (clothing, helmet) instead of body color.

Attributes in OPTIONAL only appear in the caption when their confidence reaches --min_conf, so
unreliable guesses (helmet color, scooter vs underbone from behind) do not pollute captions.

    python -m CLIP_Query.generate_text_labels
    python -m CLIP_Query.generate_text_labels --model ViT-L-14 --pretrained laion2b_s32b_b82k
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Which image folder each split's images live in
SPLIT_DIRS = {"train": "train", "val": "train", "test": "test"}

VEHICLE_TYPES = ["car", "motorcycle", "truck", "bus"]

COLORS = ["white", "black", "silver", "gray", "red", "blue", "yellow", "green", "brown", "orange"]
CLOTHING_COLORS = ["white", "black", "gray", "red", "blue", "yellow", "green", "brown", "orange", "pink"]


def _colors(template, colors=COLORS):
    return {c: template.format(c) for c in colors}


# type -> attribute -> {label: zero-shot phrase}. Phrases are wrapped by ClipEncoder's
# templates ("a photo of a {}."), so they start without an article.
ATTRIBUTES = {
    "car": {
        "color": _colors("{} car"),
        "style": {"sedan": "sedan car", "SUV": "SUV", "hatchback": "hatchback car",
                  "pickup": "pickup truck", "minivan": "minivan"},
    },
    "motorcycle": {
        "style": {"scooter": "scooter", "underbone": "underbone motorbike",
                  "sport": "sport bike", "standard": "standard motorcycle"},
        "rider_color": _colors("motorcyclist wearing {} clothes", CLOTHING_COLORS),
        "helmet": _colors("{} helmet", CLOTHING_COLORS),
    },
    "truck": {
        "color": _colors("{} truck"),
        "style": {"box": "box truck", "flatbed": "flatbed truck", "dump": "dump truck",
                  "tanker": "tanker truck", "container": "container truck"},
    },
    "bus": {
        "color": _colors("{} bus"),
        "style": {"city": "city bus", "mini": "minibus", "coach": "coach bus"},
    },
}

# Attributes scored on a sub-crop instead of the whole image: (top, bottom, left, right) fractions.
# The helmet is at the top centre of the rear-view crops; on the full image CLIP just repeats the
# clothing color.
CROPS = {("motorcycle", "helmet"): (0.0, 0.3, 0.2, 0.8)}

OPTIONAL = {"motorcycle": {"style", "helmet"}}

# Optional attributes CLIP cannot judge (scooter vs underbone from behind: the cameras agree on only
# about half the vehicles); they only reach the caption when set by hand in text_labels_manual.json
MANUAL_ONLY = {"motorcycle": {"style"}}

# The attribute that is checked for agreement between cameras
MAIN_COLOR = {"car": "color", "motorcycle": "rider_color", "truck": "color", "bus": "color"}

VIEWS = {"front": "{} seen from the front", "rear": "{} seen from behind", "side": "{} seen from the side"}

LOGIT_SCALE = 100.0  # CLIP's trained temperature


def kept_attributes(vehicle_type, attrs, confs, min_conf, keep=None):
    """Attributes that are reliable enough for the caption: confident, the cameras agree (keep marks
    False otherwise) and, for MANUAL_ONLY ones, set by hand. Hand-set values have confidence 1.0."""
    keep = keep or {}
    return {name: value for name, value in attrs.items()
            if confs[name] >= min_conf and keep.get(name, True)}


def make_caption(vehicle_type, attrs, confs, min_conf, keep=None):
    """Builds the caption from the reliable attributes only; unreliable ones are left out, so a
    vehicle whose cameras disagree on the color still gets e.g. "SUV car" instead of a wrong color."""
    kept = kept_attributes(vehicle_type, attrs, confs, min_conf, keep)
    if vehicle_type == "motorcycle":
        caption = f"{kept['style']} motorcycle" if "style" in kept else "motorcycle"
        if "rider_color" in kept and "helmet" in kept:
            caption += f", rider in {kept['rider_color']} clothes and a {kept['helmet']} helmet"
        elif "rider_color" in kept:
            caption += f", rider in {kept['rider_color']} clothes"
        elif "helmet" in kept:
            caption += f", rider with a {kept['helmet']} helmet"
        return caption
    return " ".join([kept[name] for name in ("color", "style") if name in kept] + [vehicle_type])


def is_informative(vehicle_type, attrs, confs, min_conf, keep=None):
    """False when the caption says nothing beyond the vehicle type (or, for motorcycles, nothing
    about the rider): such a vehicle is flagged for review and has no usable text label."""
    kept = kept_attributes(vehicle_type, attrs, confs, min_conf, keep)
    names = ("rider_color", "helmet") if vehicle_type == "motorcycle" else ("color", "style")
    return any(name in kept for name in names)


def query_phrases(vehicle):
    """Evaluation queries for one labelled vehicle: full caption, color only, style only.
    style is None when it was left out of the caption (no reliable value to query for)."""
    attrs, vehicle_type = vehicle["attributes"], vehicle["type"]
    if vehicle_type == "motorcycle":
        color = f"motorcyclist wearing {attrs['rider_color']} clothes"
    else:
        color = f"{attrs['color']} {vehicle_type}"
    style = f"{attrs['style']} {vehicle_type}" if vehicle.get("keep", {}).get("style", True) else None
    return {"caption": vehicle["caption"], "color": color, "style": style}


def crop(image, box):
    top, bottom, left, right = box
    h, w = image.shape[:2]
    return image[int(h * top):max(int(h * bottom), int(h * top) + 1),
                 int(w * left):max(int(w * right), int(w * left) + 1)]


def parse_name(img_name):
    """car_0088_cam2.jpg -> ('car_0088', 'car', 2)"""
    vehicle_type, vehicle_id, cam = Path(img_name).stem.split("_")
    return f"{vehicle_type}_{vehicle_id}", vehicle_type, int(cam[3:])


def read_split(data_dir: Path, split: str):
    with open(data_dir / f"{split}.txt") as f:
        return [line.strip() for line in f if line.strip()]


def text_bank(encoder: 'ClipEncoder', phrases):
    return np.stack([encoder.encode_text(p) for p in phrases])


def softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def zero_shot(image_feats, bank):
    return softmax(LOGIT_SCALE * image_feats @ bank.T)


def generate(encoder: 'ClipEncoder', data_dir: Path, min_conf: float):
    entries = []  # (img_name, split, path)
    for split, folder in SPLIT_DIRS.items():
        for img_name in read_split(data_dir, split):
            entries.append((img_name, split, data_dir / "images" / folder / img_name))

    images = [cv2.imread(str(path)) for _, _, path in entries]
    missing = [str(path) for (_, _, path), img in zip(entries, images) if img is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images could not be read, e.g. {missing[:3]}")

    print(f"[Labels] Embedding {len(images)} images")
    feats = encoder.encode_images(images)
    crop_feats = {}  # (type, attribute) -> {entry index: feature}
    for (vehicle_type, name), box in CROPS.items():
        idx = [i for i, (img_name, _, _) in enumerate(entries) if parse_name(img_name)[1] == vehicle_type]
        crop_feats[(vehicle_type, name)] = dict(zip(idx, encoder.encode_images([crop(images[i], box) for i in idx])))

    type_probs = zero_shot(feats, text_bank(encoder, VEHICLE_TYPES))
    banks = {t: {name: text_bank(encoder, list(options.values())) for name, options in attrs.items()}
             for t, attrs in ATTRIBUTES.items()}
    view_banks = {t: text_bank(encoder, [v.format(t) for v in VIEWS.values()]) for t in VEHICLE_TYPES}
    view_names = list(VIEWS)

    per_vehicle = defaultdict(list)
    for i, (img_name, split, _) in enumerate(entries):
        key, vehicle_type, cam = parse_name(img_name)
        feat = feats[i:i + 1]
        per_vehicle[key].append({
            "img_name": img_name, "split": split, "type": vehicle_type, "cam": cam,
            "clip_type": VEHICLE_TYPES[type_probs[i].argmax()],
            "view": view_names[zero_shot(feat, view_banks[vehicle_type])[0].argmax()],
            "probs": {name: zero_shot(crop_feats[(vehicle_type, name)][i][None] if (vehicle_type, name) in CROPS
                                      else feat, bank)[0]
                      for name, bank in banks[vehicle_type].items()},
        })

    labels = {}
    for key in sorted(per_vehicle):
        items = per_vehicle[key]
        vehicle_type = items[0]["type"]
        options = {name: list(opts) for name, opts in ATTRIBUTES[vehicle_type].items()}

        attrs, confs = {}, {}
        for name, names in options.items():
            probs = np.mean([it["probs"][name] for it in items], axis=0)
            attrs[name], confs[name] = names[probs.argmax()], round(float(probs.max()), 4)

        image_info = {}
        for it in items:
            info = {"split": it["split"], "cam": it["cam"], "view": it["view"], "clip_type": it["clip_type"]}
            for name, names in options.items():
                info[name] = names[it["probs"][name].argmax()]
                info[f"{name}_conf"] = round(float(it["probs"][name].max()), 4)
            image_info[it["img_name"]] = info

        # Averaged probabilities can be confident while the two cameras disagree, so optional
        # attributes are also dropped on disagreement
        agree = {name: len({info[name] for info in image_info.values()}) == 1 for name in options}
        keep = {name: agree[name] and name not in MANUAL_ONLY.get(vehicle_type, ()) for name in options}
        labels[key] = {
            "type": vehicle_type,
            "split": items[0]["split"],
            "attributes": attrs,
            "confidence": confs,
            "keep": keep,
            "caption": make_caption(vehicle_type, attrs, confs, min_conf, keep),
            # attributes below min_conf or without agreement are already dropped from the caption
            "needs_review": not is_informative(vehicle_type, attrs, confs, min_conf, keep),
            "manual": False,
            "images": image_info,
        }
    return labels


def apply_manual(labels, manual, min_conf):
    """Hand-checked attributes (data/text_labels_manual.json) override the CLIP guesses.

    manual: vehicle key -> {attribute: value}; a value of null leaves that optional attribute out
    of the caption (it cannot be told from the images).
    """
    for key, fixed in manual.items():
        if key.startswith("_"):  # comments
            continue
        v = labels[key]
        spec = ATTRIBUTES[v["type"]]
        for name, value in fixed.items():
            if value is None:
                if name not in OPTIONAL.get(v["type"], ()):
                    raise ValueError(f"{key}: only optional attributes can be null, not {name}")
                v["keep"][name] = False
            else:
                if value not in spec[name]:
                    raise ValueError(f"{key}: {name}={value!r} is not one of {list(spec[name])}")
                v["attributes"][name] = value
                v["confidence"][name] = 1.0
                v["keep"][name] = True
        v["caption"] = make_caption(v["type"], v["attributes"], v["confidence"], min_conf, v["keep"])
        v["needs_review"] = not is_informative(v["type"], v["attributes"], v["confidence"], min_conf, v["keep"])
        v["manual"] = True
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate text labels for the Re-ID dataset with CLIP")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default=None, help="Defaults to <data_dir>/text_labels.json")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--min_conf", type=float, default=0.4,
                        help="Below this confidence an optional attribute is left out of the caption "
                             "and a required one flags the vehicle for review")
    parser.add_argument("--recaption", action="store_true",
                        help="Rebuild captions and flags from the existing --out file without running CLIP")
    parser.add_argument("--manual", default=None,
                        help="Hand-checked attributes that override CLIP; defaults to <data_dir>/text_labels_manual.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "text_labels.json"
    if args.recaption:
        with open(out) as f:
            labels = json.load(f)["vehicles"]
        for v in labels.values():
            v["caption"] = make_caption(v["type"], v["attributes"], v["confidence"], args.min_conf, v["keep"])
            v["needs_review"] = not is_informative(v["type"], v["attributes"], v["confidence"],
                                                   args.min_conf, v["keep"])
    else:
        from CLIP_Query.clip_encoder import ClipEncoder
        labels = generate(ClipEncoder(args.model, args.pretrained), data_dir, args.min_conf)
    manual_path = Path(args.manual) if args.manual else data_dir / "text_labels_manual.json"
    if manual_path.exists():
        with open(manual_path) as f:
            manual = json.load(f)
        apply_manual(labels, manual, args.min_conf)
        print(f"[Labels] Applied {sum(not k.startswith('_') for k in manual)} hand-checked vehicles from {manual_path}")

    with open(out, "w") as f:
        json.dump({"meta": {"model": args.model, "pretrained": args.pretrained, "min_conf": args.min_conf},
                   "vehicles": labels}, f, indent=2)
    flagged = sum(v["needs_review"] for v in labels.values())
    print(f"[Labels] Wrote {len(labels)} vehicles to {out} ({flagged} flagged for review)")


if __name__ == "__main__":
    main()
