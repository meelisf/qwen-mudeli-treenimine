#!/usr/bin/env python3
"""Kiire test: Kurrent mudel ühe pildi peal."""
import sys
from PIL import Image
from unsloth import FastVisionModel
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("prompt", pathlib.Path(__file__).parent / "prompt.py")
prompt_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prompt_mod)
KURRENT_INSTRUCTION = prompt_mod.KURRENT_INSTRUCTION

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/test/PXL_20260601_083026114.MP.jpg"
model_path = sys.argv[2] if len(sys.argv) > 2 else "models/qwen3.5-ocr-kurrent-20260601"

print(f"Laadimine: {model_path}")
model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

tokenizer.image_processor.size = {'longest_edge': 5_120_000, 'shortest_edge': 65536}

print(f"Pilt: {image_path}")
image = Image.open(image_path).convert("RGB")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": KURRENT_INSTRUCTION},
]}]

text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text=[text], images=[image], return_tensors="pt").to("cuda")

print("Genereerin...\n")
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.1,
    do_sample=False,
)
decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("="*60)
print(decoded)
print("="*60)

import pathlib
img_stem = pathlib.Path(image_path).stem
model_name = pathlib.Path(model_path).name
out_dir = pathlib.Path("data/test/model-outputs") / model_name
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{img_stem}.txt"
out_path.write_text(decoded, encoding="utf-8")
print(f"Salvestatud: {out_path}")
