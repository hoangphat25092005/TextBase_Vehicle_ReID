"""Generate per-vehicle text labels with an open vision-language model (Qwen2.5-VL).

Both camera images of a vehicle go into one prompt and the model answers with a JSON object over the
same attribute vocabulary as the CLIP labels (ATTRIBUTES in generate_text_labels.py), so captions,
the dataset class and the trainer work unchanged. An attribute the model cannot tell is "unknown":
it gets confidence 0 and is left out of the caption.

Val/test vehicles have hand-checked attributes in data/text_labels.json, so the script also prints the
VLM accuracy per attribute next to the CLIP accuracy per image.

    python -m CLIP_Query.generate_vlm_labels --splits val test
    python -m CLIP_Query.generate_vlm_labels --splits val test train --model Qwen/Qwen2.5-VL-7B-Instruct --load_4bit

Raw answers are cached in <out>.raw.jsonl, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import re
from pathlib import Path

from CLIP_Query.generate_text_labels import ATTRIBUTES, SPLIT_DIRS, is_informative, make_caption

ROOT = Path(__file__).resolve().parents[1]

DESCRIBE = {
    "color": "the main body color of the {type}",
    "style": "the body style of the {type}",
    "rider_color": "the main color of the rider's clothes (jacket or shirt)",
    "helmet": "the color of the rider's helmet",
}

STYLE_HINTS = {
    "motorcycle": "scooter = step-through frame with a flat footboard; underbone = small step-through "
                  "frame with a visible backbone tube and exposed engine; sport = fully faired fast bike; "
                  "standard = upright naked bike",
}


def build_prompt(vehicle_type, n_images):
    lines = []
    for name, options in ATTRIBUTES[vehicle_type].items():
        line = (f'- "{name}": {DESCRIBE[name].format(type=vehicle_type)}. '
                f'One of: {", ".join(options)}, or "unknown" if the photos do not show it.')
        if name == "style" and vehicle_type in STYLE_HINTS:
            line += f" ({STYLE_HINTS[vehicle_type]})"
        lines.append(line)
    seen = ("The two photos show the same vehicle from two different cameras" if n_images > 1
            else "The photo shows a vehicle")
    note = (" Motorcycles are often seen from behind, with the rider covering the bike." if vehicle_type == "motorcycle"
            else "")
    return (f"{seen}; it is a {vehicle_type}.{note} Describe it by answering with one JSON object and nothing "
            f"else, using exactly these keys:\n" + "\n".join(lines))


def parse_answer(text, vehicle_type):
    """Model text -> {attribute: option or None}. Anything that is not a listed option becomes None."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    try:
        raw = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        raw = {}
    answer = {}
    for name, options in ATTRIBUTES[vehicle_type].items():
        value = str(raw.get(name, "")).strip().lower()
        answer[name] = next((o for o in options if o.lower() == value), None)
    return answer


def load_model(name, load_4bit):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs).eval()
    # The crops are small; capping the pixels keeps memory low on a T4
    processor = AutoProcessor.from_pretrained(name, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
    return model, processor


def ask(bundle, images, prompt, max_new_tokens=100):
    import torch

    model, processor = bundle
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def vlm_vehicle(base, answer, min_conf):
    """Label entry for one vehicle from the VLM answer; unknown attributes are dropped from the caption."""
    vehicle_type = base["type"]
    attrs, confs, keep = {}, {}, {}
    for name, options in ATTRIBUTES[vehicle_type].items():
        value = answer[name]
        attrs[name] = value if value else next(iter(options))  # placeholder, never used in the caption
        confs[name] = 1.0 if value else 0.0
        keep[name] = value is not None
    return {
        "type": vehicle_type, "split": base["split"], "attributes": attrs, "confidence": confs, "keep": keep,
        "caption": make_caption(vehicle_type, attrs, confs, min_conf, keep),
        "needs_review": not is_informative(vehicle_type, attrs, confs, min_conf, keep),
        "manual": False, "images": base["images"],
    }


def evaluate(base_vehicles, answers):
    """Accuracy per (type, attribute) against the hand-checked vehicles. Attributes the hand check left
    out (keep False) are skipped. CLIP is scored per image, the VLM per vehicle (it sees both images)."""
    rows = {}
    for key, answer in answers.items():
        base = base_vehicles[key]
        if not base["manual"]:
            continue
        for name, truth in base["attributes"].items():
            if not base["keep"][name]:
                continue
            row = rows.setdefault((base["type"], name), {"n": 0, "vlm": 0, "answered": 0, "clip": 0, "images": 0})
            row["n"] += 1
            row["answered"] += answer[name] is not None
            row["vlm"] += answer[name] == truth
            for info in base["images"].values():
                row["images"] += 1
                row["clip"] += info[name] == truth
    lines = [f"{'type':<11}{'attribute':<13}{'n':>4}  {'VLM acc':>8}  {'answered':>8}  {'CLIP acc (per image)':>21}"]
    for (vehicle_type, name), r in sorted(rows.items()):
        lines.append(f"{vehicle_type:<11}{name:<13}{r['n']:>4}  {r['vlm'] / r['n']:>8.3f}  "
                     f"{r['answered'] / r['n']:>8.3f}  {r['clip'] / r['images']:>21.3f}")
    return lines, rows


def caption_match(base_vehicles, labels):
    pairs = [(labels[k]["caption"], base_vehicles[k]["caption"]) for k in labels if base_vehicles[k]["manual"]]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan"), len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Generate text labels with an open VLM")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--base", default=None, help="CLIP labels (vehicle list, hand-checked truth); "
                                                     "defaults to <data_dir>/text_labels.json")
    parser.add_argument("--out", default=None, help="Defaults to <data_dir>/text_labels_vlm.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--load_4bit", action="store_true", help="4-bit weights; needed for the 7B model on a T4")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=list(SPLIT_DIRS))
    parser.add_argument("--min_conf", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N vehicles (quick test)")
    args = parser.parse_args()

    from PIL import Image

    data_dir = Path(args.data_dir)
    base_path = Path(args.base) if args.base else data_dir / "text_labels.json"
    out = Path(args.out) if args.out else data_dir / "text_labels_vlm.json"
    raw_path = out.with_suffix(".raw.jsonl")
    with open(base_path) as f:
        base_file = json.load(f)
    base_vehicles = base_file["vehicles"]

    keys = [k for k in sorted(base_vehicles) if base_vehicles[k]["split"] in args.splits][:args.limit]
    raw = {}
    if raw_path.exists():
        with open(raw_path) as f:
            raw = {row["key"]: row["text"] for row in map(json.loads, f)}
    todo = [k for k in keys if k not in raw]
    print(f"[VLM] {len(keys)} vehicles in {args.splits}, {len(raw)} cached, {len(todo)} to run")

    if todo:
        bundle = load_model(args.model, args.load_4bit)
        with open(raw_path, "a") as f:
            for i, key in enumerate(todo, 1):
                base = base_vehicles[key]
                names = sorted(base["images"], key=lambda n: base["images"][n]["cam"])
                images = [Image.open(data_dir / "images" / SPLIT_DIRS[base["split"]] / n).convert("RGB") for n in names]
                text = ask(bundle, images, build_prompt(base["type"], len(images)))
                raw[key] = text
                f.write(json.dumps({"key": key, "text": text}) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"[VLM] {i}/{len(todo)}  {key}: {text.strip()[:100]!r}")

    answers = {k: parse_answer(raw[k], base_vehicles[k]["type"]) for k in keys}
    labels = {k: vlm_vehicle(base_vehicles[k], answers[k], args.min_conf) for k in keys}
    with open(out, "w") as f:
        # model / pretrained stay those of the CLIP text encoder the trainer embeds the captions with
        json.dump({"meta": {**base_file["meta"], "min_conf": args.min_conf, "labeler": args.model,
                            "splits": args.splits},
                   "vehicles": labels}, f, indent=2)
    unparsable = sum(all(v is None for v in a.values()) for a in answers.values())
    flagged = sum(v["needs_review"] for v in labels.values())
    print(f"[VLM] Wrote {len(labels)} vehicles to {out} ({flagged} flagged, {unparsable} unparsable answers)")

    lines, _ = evaluate(base_vehicles, answers)
    if len(lines) > 1:
        print("\n[Eval] against hand-checked val/test attributes")
        print("\n".join(lines))
        rate, n = caption_match(base_vehicles, labels)
        print(f"\nFull caption identical to the hand-checked caption: {rate:.3f} over {n} vehicles")


if __name__ == "__main__":
    main()
