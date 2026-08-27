#!/usr/bin/env python3
"""
LoRA adapteri liitmine baasmudelisse (merged BF16 checkpoint)

Vaja siis, kui mudelit tahetakse viia välja unslothi/transformersi maailmast:
GGUF (llama.cpp) ja NInferi converter tahavad mõlemad tervet mudelit, mitte
adapterit.

Käib **CPU peal** (~18 GB RAM), et GPU jääks OCR-teenusele vabaks.

  python scripts/merge_lora.py models/qwen3.5-ocr-kurrent-20260602
  python scripts/merge_lora.py <adapter> --out models/merged/<nimi>

NB! Tokenizer/preprocessor failid kopeeritakse **ametlikust baasmudelist**,
mitte meie checkpointist. Meie treeningkood kirjutab `processor_config.json`-i
muudetud pildieelarve (5,12M px ametliku 16M asemel) ja NInferi converter
kontrollib nende failide SHA256-i ametlike vastu. llama.cpp seda ei kontrolli,
aga hoiame ahela ühesugusena.
"""

import shutil
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText

FRONTEND_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json",
]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    print(__doc__)
    sys.exit(1)

ADAPTER = Path(args[0])
OUT = None
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT = Path(sys.argv[i + 1])
    elif a.startswith("--out="):
        OUT = Path(a.split("=", 1)[1])
if OUT is None:
    OUT = Path("models/merged") / f"{ADAPTER.name}-bf16"

if not ADAPTER.exists():
    print(f"Viga: adapterit ei leitud: {ADAPTER}")
    sys.exit(1)

peft_cfg = PeftConfig.from_pretrained(ADAPTER)
BASE = peft_cfg.base_model_name_or_path
print(f"Adapter:  {ADAPTER}")
print(f"Baas:     {BASE}")
print(f"Väljund:  {OUT}")
print(f"LoRA:     r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}")

print("\nLaen baasmudeli (CPU, bf16)...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cpu",
)

print("Lisan adapteri...")
model = PeftModel.from_pretrained(model, str(ADAPTER), device_map="cpu")

print("Liidan kaalud (merge_and_unload)...")
model = model.merge_and_unload()

OUT.mkdir(parents=True, exist_ok=True)
print(f"Salvestan: {OUT}")
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")

# Ametlikud frontend-failid baasmudeli snapshotist
from huggingface_hub import snapshot_download
base_dir = Path(BASE) if Path(BASE).exists() else Path(
    snapshot_download(BASE, allow_patterns=FRONTEND_FILES)
)
kopeeritud = []
for nimi in FRONTEND_FILES:
    src = base_dir / nimi
    if src.exists():
        shutil.copy2(src, OUT / nimi)
        kopeeritud.append(nimi)
print(f"Frontend-failid ametlikust baasist: {', '.join(kopeeritud)}")

kokku = sum(f.stat().st_size for f in OUT.glob("*")) / 1024**3
print(f"\nValmis: {OUT} ({kokku:.1f} GB)")
print("Järgmine samm:")
print(f"  python ~/Dokumendid/LLM/llama.cpp/convert_hf_to_gguf.py {OUT} --outtype bf16")
print(f"  python ~/Dokumendid/LLM/llama.cpp/convert_hf_to_gguf.py {OUT} --mmproj --outtype f16")
