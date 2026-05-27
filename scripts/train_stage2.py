#!/usr/bin/env python3
"""
Qwen3.5-9B OCR peenhäälestus – ETAPP 2

Andmed:   data/processed/ (136 lehekülge, täismärgendus)
Mudel:    models/qwen3.5-ocr-lora/ (etapp 1 checkpoint – LoRA juba küljes!)
Väljund:  models/qwen3.5-ocr-lora-stage2/

Käivitamine:
  python scripts/train_stage2.py          # täistreening
  python scripts/train_stage2.py --test   # kiirtest, 5 sammu, ei salvestata

Eeldused:
  - Etapp 1 treening on lõpetatud (models/qwen3.5-ocr-lora/ olemas)
  - data/processed/metadata.csv on loodud (käivita scripts/prepare_raw.py)

Erinevus etapp 1 skriptist:
  - get_peft_model() EI kutsuta – etapp 1 checkpointis on LoRA juba küljes.
    Uuesti kutsumine annab RuntimeError "You already added LoRA adapters".
    Checkpoint laaditakse otse treenimiseks valmis olekusse.
"""

import os
import sys
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

DATA_ROOT_DIR = "data/processed"
CSV_PATH      = os.path.join(DATA_ROOT_DIR, "metadata.csv")
IMAGES_DIR    = os.path.join(DATA_ROOT_DIR, "images")
MODEL_NAME    = "models/qwen3.5-ocr-lora"   # etapp 1 checkpoint
OUTPUT_PATH   = "models/qwen3.5-ocr-lora-stage2"

print(f"Andmestik: {DATA_ROOT_DIR}")
print(f"Lähtepunkt: {MODEL_NAME}")
print(f"Salvestuskoht: {OUTPUT_PATH}")

if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(
        f"Etapp 1 checkpoint ei leitud: {MODEL_NAME}\n"
        "Käivita esmalt: python scripts/train.py"
    )
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Andmestiku faili ei leitud: {CSV_PATH}\n"
        "Käivita esmalt: python scripts/prepare_raw.py"
    )
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Piltide kausta ei leitud: {IMAGES_DIR}")

# ---------------------------------------------------------------------------
# Mudeli laadimine etapp 1 checkpointist
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

# Laadib baasmodeli + rakendab etapp 1 LoRA adapterid.
# LoRA adapterid on juba küljes – get_peft_model() EI tohi järgneda!
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
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

# get_peft_model() välja jäetud – checkpoint sisaldab juba LoRA adaptereid
print("Mudel laaditud etapp 1 checkpointist (LoRA adapterid juba küljes).")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Andmestiku laadimine
# ---------------------------------------------------------------------------

dataset = load_dataset("csv", data_files=CSV_PATH, split="train")

def resolve_path(example):
    example["failinimi"] = os.path.join(IMAGES_DIR, os.path.basename(example["failinimi"]))
    return example

dataset = dataset.map(resolve_path)
print(f"Andmestik laaditud: {len(dataset)} näidet.")

# ---------------------------------------------------------------------------
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
        warmup_steps=2 if TEST_MODE else 5,
        max_steps=5 if TEST_MODE else -1,
        num_train_epochs=1 if TEST_MODE else 2,
        learning_rate=1e-4,      # etapp 2: väiksem LR kui etapp 1 (2e-4), et mitte üle kirjutada
        logging_steps=1 if TEST_MODE else 5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="models/checkpoints-stage2",
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
