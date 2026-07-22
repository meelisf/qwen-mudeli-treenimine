#!/usr/bin/env python3
"""
Qwen3.5-9B OCR markup-treenimine – inkrementaalne

Jätkab etapp 1 checkpoindist ja treenib markup-reegleid:
  - kursiivimarkeering (*tekst*)
  - rasvane kiri (**tekst**)
  - koodivahetuse markeering (~tekst~)
  - ääremärkused ([[m: sisu]])

Andmestik (kombineeritud):
  - data/processed/   – 136 lk, vanem käsitsi märgendatud materjal
  - data/vutt/        – VUTT-ist sünkroniseeritud Valmis leheküljed

Käivitamine:
  python scripts/train_markup.py           # täistreening
  python scripts/train_markup.py --test    # kiirtest, 5 sammu, ei salvestata
  python scripts/train_markup.py --base models/qwen3.5-ocr-lora-v2   # muu checkpoint
"""

import os
import sys
import csv
import torch
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image as PILImage
from prompt import INSTRUCTION
from convert_marginalia import clean_markup
from imaging import MAX_PIXELS

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print("*** TESTREŽIIM: max 5 sammu, ei salvestata ***")

# ---------------------------------------------------------------------------
# Konfid
# ---------------------------------------------------------------------------

# Etapp 1 checkpoint (baas) – backup enne markup treeningut
BASE_MODEL = "models/qwen3.5-ocr-lora-backup-20260527"
for arg in sys.argv:
    if arg.startswith("--base="):
        BASE_MODEL = arg.split("=", 1)[1]
    elif arg == "--base" and sys.argv.index(arg) + 1 < len(sys.argv):
        BASE_MODEL = sys.argv[sys.argv.index(arg) + 1]

# Andmeallikad – ainult VUTT märgendatud materjal
DATA_SOURCES = [
    {"csv": "data/vutt/metadata.csv", "images": "data/vutt/images"},
]

# Kuupäevaga väljundkausta nimi
DATE_STAMP  = datetime.now().strftime("%Y%m%d")
OUTPUT_PATH = f"models/qwen3.5-ocr-markup-{DATE_STAMP}"
CKPT_DIR    = f"models/checkpoints-markup-{DATE_STAMP}"

print(f"Lähtepunkt:   {BASE_MODEL}")
print(f"Salvestuskoht: {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Eelkontrollid
# ---------------------------------------------------------------------------

if not Path(BASE_MODEL).exists():
    print(f"Viga: checkpoint ei leitud: {BASE_MODEL}")
    print("Käivita esmalt: python scripts/train.py")
    sys.exit(1)

missing_sources = []
for src in DATA_SOURCES:
    if not Path(src["csv"]).exists() or not Path(src["images"]).exists():
        missing_sources.append(src["csv"])

if missing_sources:
    print("Hoiatus: järgmised andmeallikad puuduvad ja jäetakse vahele:")
    for s in missing_sources:
        print(f"  {s}")

available_sources = [
    src for src in DATA_SOURCES
    if Path(src["csv"]).exists() and Path(src["images"]).exists()
]
if not available_sources:
    print("Viga: ühtki andmeallikat ei leitud.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mudeli laadimine
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

tokenizer.truncation = False

tokenizer.image_processor.size = {
    "longest_edge": MAX_PIXELS,
    "shortest_edge": tokenizer.image_processor.size.get("shortest_edge", 65536),
}
print(f"Pildi max_pixels: {MAX_PIXELS:,} px "
      f"→ ~{MAX_PIXELS // 1024} visuaaltokenit")

# Checkpoint sisaldab juba LoRA adaptereid – get_peft_model() EI tohi järgneda
print("Mudel laaditud etapp 1 checkpoindist (LoRA adapterid juba küljes).")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Andmestiku laadimine mitmest allikast
# ---------------------------------------------------------------------------

class LehekyljAndmestik:
    """Laisk laadimine mitmest CSV-st."""

    def __init__(self, sources):
        self.samples = []
        skipped = 0
        cleaned_m = 0

        for src in sources:
            csv_path   = src["csv"]
            images_dir = src["images"]

            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get("transkriptsioon", "")
                    if not isinstance(t, str) or not t.strip():
                        skipped += 1
                        continue
                    # Sama puhastusahel ka siin: olemasolevad metadata.csv-d on
                    # ehitatud enne nende sammude lisamist ja sisaldavad veel
                    # <annN> tage ning ristuvat pesastust.
                    t_clean = clean_markup(t)
                    if not t_clean:
                        skipped += 1
                        continue
                    if t_clean != t.strip():
                        cleaned_m += 1
                    img_path = os.path.join(images_dir, os.path.basename(row["failinimi"]))
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    self.samples.append({
                        "failinimi": img_path,
                        "transkriptsioon": t_clean,
                    })

        if skipped:
            print(f"  Hoiatus: {skipped} rida jäeti vahele.")
        if cleaned_m:
            print(f"  Normaliseeritud/puhastatud markup: {cleaned_m} leheküljel.")

        print(f"  Andmestik: {len(self.samples)} näidet ({len(sources)} allikast)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image", "image": s["failinimi"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": s["transkriptsioon"]}],
                },
            ]
        }


converted_dataset = LehekyljAndmestik(available_sources)

if len(converted_dataset) == 0:
    print("Viga: andmestik on tühi.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Treener
# ---------------------------------------------------------------------------

FastVisionModel.for_training(model)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu_stats.name}, mälu: {round(gpu_stats.total_memory / 1024**3, 1)} GB")
print(f"Treeninguandmeid kokku: {len(converted_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=converted_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max", max_seq_length=8192),
    args=SFTConfig(
        max_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if TEST_MODE else 8,
        warmup_steps=2 if TEST_MODE else 10,
        max_steps=5 if TEST_MODE else -1,
        num_train_epochs=1 if TEST_MODE else 2,
        learning_rate=1e-4,     # inkrementaalne: väiksem LR kui etapp 1 (2e-4)
        logging_steps=1 if TEST_MODE else 10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=CKPT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0,
    ),
)

print("Alustan treeningut...")
trainer_stats = trainer.train()
print(f"Treening lõppenud! Aeg: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} min")

# ---------------------------------------------------------------------------
# Salvestamine
# ---------------------------------------------------------------------------

if TEST_MODE:
    print("Testrežiim: salvestamine vahele jäetud.")
else:
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"LoRA adapter salvestatud: {OUTPUT_PATH}")
    print(f"\nKui tulemus rahuldab, aktiveeri mudel:")
    print(f"  sudo systemctl stop ocr-service")
    print(f"  cp -r {OUTPUT_PATH} models/qwen3.5-ocr-lora-backup-{DATE_STAMP}")
    print(f"  rm -rf models/qwen3.5-ocr-lora && cp -r {OUTPUT_PATH} models/qwen3.5-ocr-lora")
    print(f"  sudo systemctl start ocr-service")

# ---------------------------------------------------------------------------
# Testimine esimese pildiga
# ---------------------------------------------------------------------------

print("\nTestan treenitud mudelit...")
FastVisionModel.for_inference(model)

first_src = available_sources[0]
test_images = sorted(
    f for f in Path(first_src["images"]).iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if not test_images:
    print("Testpilte ei leitud.")
else:
    test_image_path = test_images[0]
    image = PILImage.open(test_image_path)
    print(f"Testpilt: {test_image_path}")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image"},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=True, do_sample=False)
    import re as _re
    raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    print("\n--- TULEMUS ---")
    print(raw)
