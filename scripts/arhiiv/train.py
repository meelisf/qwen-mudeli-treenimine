#!/usr/bin/env python3
"""
Qwen3.5-9B OCR peenhäälestus – ETAPP 1

Andmed:   data/lehekyljed/ (1500 lehekülge, puhas transkriptsioon)
Mudel:    unsloth/Qwen3.5-9B (baasmudelilt, LoRA lisatakse siit)
Väljund:  models/qwen3.5-ocr-lora/

Käivitamine:
  python scripts/train.py          # täistreening (~2h)
  python scripts/train.py --test   # kiirtest, 5 sammu, ei salvestata

Märkus: etapp 1 andmed sisaldavad puhast transkriptsiooni (ſ säilitatud,
¬ poolitusmärk), ilma kursiivimarkeeringute vm vormingumärkideta.
Vormingureeglitest sisaldab instruktsioon kõiki reegleid, kuid etapp 1
näited demonstreerivad ainult neid, mis andmetes tegelikult esinevad.
Etapp 2 (train_stage2.py) õpetab täismärgenduse konkreetsete näidetega.
"""

import os
import sys
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Vähendab CUDA mälu fragmentatsiooni (aitab Triton OOM vastu)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image as PILImage
from prompt import INSTRUCTION

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print("*** TESTREŽIIM: max 5 sammu, ei salvestata ***")

# ---------------------------------------------------------------------------
# Konfid
# ---------------------------------------------------------------------------

DATA_ROOT_DIR = "data/lehekyljed"
CSV_PATH      = os.path.join(DATA_ROOT_DIR, "metadata.csv")
IMAGES_DIR    = os.path.join(DATA_ROOT_DIR, "images")
MODEL_NAME    = "unsloth/Qwen3.5-9B"
OUTPUT_PATH   = "models/qwen3.5-ocr-lora"

print(f"Andmestik: {DATA_ROOT_DIR}")
print(f"Lähtepunkt: {MODEL_NAME}")
print(f"Salvestuskoht: {OUTPUT_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Andmestiku faili ei leitud: {CSV_PATH}")
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Piltide kausta ei leitud: {IMAGES_DIR}")

# ---------------------------------------------------------------------------
# Mudeli laadimine
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Keelame kärpimise – lahendab "Mismatch in `image` token count" vea
tokenizer.truncation = False

# Piira pildi resolutsiooni treenimiseks.
# Qwen3.5-9B vaikeväärtus on 16M px → 14000+ visuaaltokenit → OOM.
# NB! longest_edge on PIKSLITE KOGUARV, mitte serva pikkus.
# 5M px ≈ 4900 visuaaltokenit (valem: visual_tokens ≈ total_pixels / 1024).
tokenizer.image_processor.size = {
    "longest_edge": 5_120_000,
    "shortest_edge": tokenizer.image_processor.size.get("shortest_edge", 65536),
}
print(f"Pildi max_pixels: {tokenizer.image_processor.size['longest_edge']:,} px "
      f"→ ~{5_120_000 // 1024} visuaaltokenit")

# Keelame reasoning/thinking tokenid – OCR-i jaoks pole vaja
tokenizer.chat_template = (
    tokenizer.chat_template.replace("enable_thinking=True", "enable_thinking=False")
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template
    else tokenizer.chat_template
)

# Baasmudelile lisatakse LoRA adapterid – treenime kõiki kihte
model = FastVisionModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
    finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
)
print("Mudel laaditud ja LoRA jaoks ette valmistatud.")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Andmestiku laadimine
# ---------------------------------------------------------------------------

dataset = load_dataset("csv", data_files=CSV_PATH, split="train")

def resolve_path(example):
    # CSV-s on teed kujul "images/failinimi.jpg"; teisendame absoluutseks
    example["failinimi"] = os.path.join(IMAGES_DIR, os.path.basename(example["failinimi"]))
    return example

dataset = dataset.map(resolve_path)
print(f"Andmestik laaditud: {len(dataset)} näidet.")



class LehekyljAndmestik:
    """Laisk laadimine: pildid loetakse mällu alles treenimise ajal."""

    def __init__(self, samples):
        valid = []
        skipped = 0
        for s in samples:
            t = s.get("transkriptsioon")
            if not isinstance(t, str) or not t.strip():
                skipped += 1
                continue
            if not os.path.exists(s["failinimi"]):
                skipped += 1
                continue
            valid.append(s)
        if skipped:
            print(f"  Hoiatus: {skipped} rida jäeti vahele (tühi transkriptsioon või puuduv fail).")
        self.samples = valid

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


converted_dataset = LehekyljAndmestik(dataset)
print(f"Andmestik valmis: {len(converted_dataset)} näidet (laisk laadimine).")

# ---------------------------------------------------------------------------
# Treener
# ---------------------------------------------------------------------------

FastVisionModel.for_training(model)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu_stats.name}, mälu: {round(gpu_stats.total_memory / 1024**3, 1)} GB")

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
        learning_rate=2e-4,
        logging_steps=1 if TEST_MODE else 10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="models/checkpoints",
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

# ---------------------------------------------------------------------------
# Testimine treeninguandmestiku esimese pildiga
# ---------------------------------------------------------------------------

print("\nTestan treenitud mudelit...")
FastVisionModel.for_inference(model)

test_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not test_images:
    print("Testpilte ei leitud, jätan testimise vahele.")
else:
    test_image_path = os.path.join(IMAGES_DIR, test_images[0])
    image = PILImage.open(test_image_path)
    print(f"Testpilt: {test_image_path}, suurus: {image.size}")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image"},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=True)
    decoded = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    print("\n--- TULEMUS ---")
    print(decoded)
