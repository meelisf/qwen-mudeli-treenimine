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

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print("*** TESTREŽIIM: max 5 sammu, ei salvestata ***")

# ---------------------------------------------------------------------------
# Konfid
# ---------------------------------------------------------------------------

# Etapp 1 checkpoint (baas)
BASE_MODEL = "models/qwen3.5-ocr-lora"
for arg in sys.argv:
    if arg.startswith("--base="):
        BASE_MODEL = arg.split("=", 1)[1]
    elif arg == "--base" and sys.argv.index(arg) + 1 < len(sys.argv):
        BASE_MODEL = sys.argv[sys.argv.index(arg) + 1]

# Andmeallikad (kasutatakse mõlemaid koos)
DATA_SOURCES = [
    {"csv": "data/processed/metadata.csv",  "images": "data/processed/images"},
    {"csv": "data/vutt/metadata.csv",        "images": "data/vutt/images"},
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
    "longest_edge": 5_120_000,
    "shortest_edge": tokenizer.image_processor.size.get("shortest_edge", 65536),
}
print(f"Pildi max_pixels: {tokenizer.image_processor.size['longest_edge']:,} px "
      f"→ ~{5_120_000 // 1024} visuaaltokenit")

tokenizer.chat_template = (
    tokenizer.chat_template.replace("enable_thinking=True", "enable_thinking=False")
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template
    else tokenizer.chat_template
)

# Checkpoint sisaldab juba LoRA adaptereid – get_peft_model() EI tohi järgneda
print("Mudel laaditud etapp 1 checkpoindist (LoRA adapterid juba küljes).")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Instruktsioon
# ---------------------------------------------------------------------------

INSTRUCTION = """You are an expert OCR assistant for historical documents.

Instructions:
1. Transcribe the entire page from the provided image.
2. Preserve original line breaks and hyphenation:
   - Antiqua hyphenation: - (regular hyphen), e.g. coa-cervare
   - Fraktur/Gothic hyphenation: \u2E17 (double hyphen), e.g. Ge\u2E17witter
3. Do not translate; keep the original language (Latin, Greek, German, Estonian, etc.).
4. Ligatures:
   - \u00E6, \u00C6, \u0153, \u0152 \u2013 transcribe exactly as they are
   - st, ff, fi, fl and other typographic ligatures \u2013 write out as separate letters
5. Umlauts and diacritics:
   - \u00F6, \u00E4, \u00FC, \u00F5 \u2013 always use modern form
   - u\u0364, o\u0364, a\u0364 (letter + superscript e) \u2013 transcribe as \u00FC, \u00F6, \u00E4
   - \u00E5, \u00C5 (Swedish) \u2013 keep as is
   - \u0169, \u00F1, \u00F5 \u2013 keep as is (tilde preserved)
6. Special characters:
   - \u017F (long s) \u2013 transcribe as \u017F
   - \u00DF (double s) \u2013 transcribe as \u00DF
7. Abbreviations:
   - que abbreviation (\uA757 etc.) \u2013 write as q;
   - -us abbreviation (\uA770) \u2013 may be expanded
8. Formatting:
   - Italic text: *between asterisks*
   - Bold text: **between double asterisks**
   - Code-switching (Fraktur word in Antiqua text or vice versa): ~between tildes~
9. Marginal notes: [[m: content of marginal note]]
10. Signature marks (quire numbers): place at the very end, e.g. A 3
11. Page breaks: if the image contains a double-page spread, mark the page break between pages with --lk--. Partial pages may occur \u2013 ignore the partial page and transcribe only where a full page is visible.

Return only the exact transcription as plain text."""

# ---------------------------------------------------------------------------
# Andmestiku laadimine mitmest allikast
# ---------------------------------------------------------------------------

class LehekyljAndmestik:
    """Laisk laadimine mitmest CSV-st."""

    def __init__(self, sources):
        self.samples = []
        skipped = 0

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
                    img_path = os.path.join(images_dir, os.path.basename(row["failinimi"]))
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    self.samples.append({
                        "failinimi": img_path,
                        "transkriptsioon": t.strip(),
                    })

        if skipped:
            print(f"  Hoiatus: {skipped} rida jäeti vahele.")

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

    outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=True)
    decoded = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    print("\n--- TULEMUS ---")
    print(decoded)
