#!/usr/bin/env python3
"""Batch test: Kurrent mudel mitme pildi peal, mudel laaditakse korra."""
import sys
import pathlib
from PIL import Image
from unsloth import FastVisionModel
import importlib.util

spec = importlib.util.spec_from_file_location("prompt", pathlib.Path(__file__).parent / "prompt.py")
prompt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_mod)
KURRENT_INSTRUCTION = prompt_mod.KURRENT_INSTRUCTION

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-ocr-kurrent-20260602"
test_dir = pathlib.Path("data/test")
gt_dir = test_dir / "ground-truth"

images = [
    test_dir / "morgenstern.jpg",
    test_dir / "1759-diarium_lk001.jpg",
    test_dir / "PXL_20260601_083026114.MP.jpg",
]

print(f"Laadimine: {model_path}")
model, tokenizer = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
FastVisionModel.for_inference(model)
tokenizer.image_processor.size = {'longest_edge': 5_120_000, 'shortest_edge': 65536}
print("Mudel laaditud.\n")

out_dir = test_dir / "model-outputs" / pathlib.Path(model_path).name
out_dir.mkdir(parents=True, exist_ok=True)

def simple_cer(hyp, ref):
    """Lihtne CER ilma editeerimiseta — char overlap / ref len."""
    import difflib
    sm = difflib.SequenceMatcher(None, hyp, ref)
    matching = sum(b.size for b in sm.get_matching_blocks())
    return 1.0 - matching / max(len(ref), 1)

for img_path in images:
    if not img_path.exists():
        print(f"PUUDUB: {img_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Pilt: {img_path.name}")

    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": KURRENT_INSTRUCTION},
    ]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text=[text], images=[image], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=False)
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    out_path = out_dir / f"{img_path.stem}.txt"
    out_path.write_text(decoded, encoding="utf-8")

    gt_path = gt_dir / f"{img_path.stem}.txt"
    if gt_path.exists():
        gt = gt_path.read_text(encoding="utf-8").strip()
        hyp = decoded.strip()
        cer = simple_cer(hyp, gt)
        print(f"CER: {cer:.1%}  (ref {len(gt)} tähemärki)")

    print("\n--- TULEMUS ---")
    print(decoded[:600])
    if len(decoded) > 600:
        print(f"... ({len(decoded)} tähemärki kokku)")
    print(f"\nSalvestatud: {out_path}")
