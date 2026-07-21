#!/usr/bin/env python3
"""
Qwen3.5 OCR mudeli testimine

Käivitamine:
  python scripts/test_model.py                          # kõik data/test/print/ pildid, aktiivne mudel
  python scripts/test_model.py data/test/pilt.jpg       # üks pilt
  python scripts/test_model.py --dir data/test/hand --model models/qwen3.5-ocr-kurrent-20260602
  python scripts/test_model.py --model models/qwen3.5-ocr-markup-20260526
  python scripts/test_model.py --model models/qwen3.5-ocr-markup-20260526 data/test/pilt.jpg

Tulemused kirjutatakse: data/test/<mudeli-nimi>/<pildi-nimi>.txt
"""

import os
import re
import sys
import torch
import gc
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel
from PIL import Image as PILImage
from prompt import INSTRUCTION

# ---------------------------------------------------------------------------
# Argumendid
# ---------------------------------------------------------------------------

MODEL_PATH = "models/qwen3.5-ocr-lora"
TEST_DIR = "data/test/print"  # print/ = trükis, hand/ = käsikiri (nagu AUTO-OCR-is)
IMAGE_PATHS = []
BATCH_SIZE = 3
THINKING = False
THINKING_BUDGET = 512  # mõtlemistokenite eelarve

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--model" and i + 1 < len(args):
        MODEL_PATH = args[i + 1]
        i += 2
    elif args[i].startswith("--model="):
        MODEL_PATH = args[i].split("=", 1)[1]
        i += 1
    elif args[i] == "--dir" and i + 1 < len(args):
        TEST_DIR = args[i + 1]
        i += 2
    elif args[i].startswith("--dir="):
        TEST_DIR = args[i].split("=", 1)[1]
        i += 1
    elif args[i] == "--thinking":
        THINKING = True
        i += 1
    elif args[i] == "--thinking-budget" and i + 1 < len(args):
        THINKING_BUDGET = int(args[i + 1])
        i += 2
    else:
        IMAGE_PATHS.append(args[i])
        i += 1

if not IMAGE_PATHS:
    test_dir = Path(TEST_DIR)
    if not test_dir.exists():
        print(f"Viga: testpiltide kaust ei leitud: {test_dir}")
        sys.exit(1)
    IMAGE_PATHS = sorted(
        str(p) for p in test_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not IMAGE_PATHS:
        print(f"Viga: {test_dir} kaustas pole pilte.")
        sys.exit(1)

if not Path(MODEL_PATH).exists():
    print(f"Viga: mudelit ei leitud: {MODEL_PATH}")
    sys.exit(1)

model_name = Path(MODEL_PATH).name
output_dir = Path("data/test") / model_name
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Mudel:    {MODEL_PATH}")
print(f"Pildid:   {len(IMAGE_PATHS)} tk")
print(f"Väljund:  {output_dir}/")

# ---------------------------------------------------------------------------
# Mudeli laadimine
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    use_gradient_checkpointing="unsloth",
)

tokenizer.truncation = False
tokenizer.image_processor.size = {
    "longest_edge": 5_120_000,
    "shortest_edge": tokenizer.image_processor.size.get("shortest_edge", 65536),
}

if not THINKING:
    tokenizer.chat_template = (
        tokenizer.chat_template.replace("enable_thinking=True", "enable_thinking=False")
        if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template
        else tokenizer.chat_template
    )

FastVisionModel.for_inference(model)
print(f"Mudel laaditud. Thinking: {'sees (budget=' + str(THINKING_BUDGET) + ')' if THINKING else 'väljas'}\n")

template_kwargs = dict(add_generation_prompt=True, tokenize=False, enable_thinking=THINKING)
if THINKING:
    template_kwargs["thinking_budget"] = THINKING_BUDGET

CHAT_TEMPLATE = tokenizer.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTION},
        {"type": "image"},
    ]}],
    **template_kwargs,
)

# ---------------------------------------------------------------------------
# Abifunktsioon
# ---------------------------------------------------------------------------

def strip_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    for marker in ["</assistant>", "<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Batch-inferents
# ---------------------------------------------------------------------------

total = len(IMAGE_PATHS)
for batch_start in range(0, total, BATCH_SIZE):
    batch_paths = IMAGE_PATHS[batch_start : batch_start + BATCH_SIZE]
    print(f"{'='*60}")
    print(f"Töötlen {batch_start+1}–{min(batch_start+BATCH_SIZE, total)} / {total}")

    images_pil = []
    valid_paths = []
    for img_path in batch_paths:
        p = Path(img_path)
        if not p.exists():
            print(f"  Viga: {img_path} ei leitud, jätan vahele.")
            continue
        try:
            img = PILImage.open(p).convert("RGB")
            images_pil.append(img)
            valid_paths.append(p)
            print(f"  {p.name}  ({img.size[0]}x{img.size[1]} px)")
        except Exception as e:
            print(f"  Viga pildi avamisel {img_path}: {e}")

    if not images_pil:
        continue

    inputs = tokenizer(
        images_pil,
        [CHAT_TEMPLATE] * len(images_pil),
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        generate_kwargs = dict(
            max_new_tokens=4096 + THINKING_BUDGET if THINKING else 4096,
            use_cache=True,
        )
        if THINKING:
            generate_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.3)
        else:
            generate_kwargs["do_sample"] = False
        outputs = model.generate(**inputs, **generate_kwargs)

    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=not THINKING)

    for img_p, raw_text in zip(valid_paths, decoded_texts):
        clean_text = strip_output(raw_text)
        out_file = output_dir / (img_p.stem + ".txt")
        out_file.write_text(clean_text, encoding="utf-8")
        print(f"\n--- {img_p.name} → {out_file} ---")
        print(clean_text)
        print()

    for img in images_pil:
        img.close()
    del inputs, outputs, decoded_texts, images_pil
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nValmis. Tulemused: {output_dir}/")
