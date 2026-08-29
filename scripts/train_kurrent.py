#!/usr/bin/env python3
"""
Qwen3.5-9B Kurrent käsikirja treenimine

Treenib käsikirja transkribeerimise oskust 12K+ lehekülge peal:
  - 19. saj saksa Kurrent (kurrent-xix)
  - 14.-17. saj Baseli piiskopkond (aaeb-xiv-xvii)
  - 16. saj Bullingeri kirjad, 130 autorit (bullinger-autoren)
  - 18. saj rootsi kohtudokumendid (svea hovratt)
  - 13.-15. saj keskaeg (koenigsfelden)

Baasmudeliks PUHAS `unsloth/Qwen3.5-9B`, LoRA r=64. NB! Mitte trükimudel –
mudel qwen3.5-ocr-kurrent-20260602 (see, mis töötab hästi) tehti puhtast
baasist. Vaikeväärtused siin vastavad sellele retseptile; ära muuda neid
ilma põhjuseta, muidu pole uus jooks vanaga võrreldav.

20260602 retsept (tõendid: models/checkpoints-kurrent-20260602/checkpoint-3178/
training_args.bin ja trainer_state.json):
  baas unsloth/Qwen3.5-9B, r=64/alpha=64, bs 1 x grad_acc 8, LR 2e-4 cosine,
  warmup 80, 2 epohhi, adamw_8bit, wd 0.01, seed 3407, bf16.
  12 712 naidet epohhis, 27,9 s/samm, epohh 12,3 h, loss 1,52 -> 0,10.
  Teine epohh andis vahe (0,12 -> 0,08), epohhi-1 checkpoint jaab alles.

Väljund: models/qwen3.5-ocr-kurrent-YYYYMMDD  (eraldi mudel, ei asenda OCR teenust)

Käivitamine:
  python scripts/train_kurrent.py           # täistreening, 20260602 retsept
  python scripts/train_kurrent.py --test    # kiirtest, 5 sammu, ei salvestata
  python scripts/train_kurrent.py --base models/muu-checkpoint
  python scripts/train_kurrent.py --resume    # jätka viimasest checkpointist
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
from transformers import TrainerCallback
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image as PILImage
from prompt import KURRENT_INSTRUCTION

TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    print("*** TESTREŽIIM: max 5 sammu, ei salvestata ***")

# ---------------------------------------------------------------------------
# Konfiguratsioon
# ---------------------------------------------------------------------------

BASE_MODEL = "unsloth/Qwen3.5-9B"   # vt docstring: 20260602 retsept
LORA_RANK = 64                      # r=16 ei mahuta 17k lk mitmesajandilist korpust
CUSTOM_STEPS = -1  # -1 = täisepohh
SAVE_STEPS = 250   # ~1,9 h (27,9 s/samm); checkpoint 1,2 GB, alles hoitakse 3
RESUME_FROM = None # --resume (viimane) või --resume=<tee checkpointini>

for i, arg in enumerate(sys.argv):
    if arg.startswith("--base="):
        BASE_MODEL = arg.split("=", 1)[1]
    elif arg == "--base" and i + 1 < len(sys.argv):
        BASE_MODEL = sys.argv[i + 1]
    elif arg.startswith("--lora-rank="):
        LORA_RANK = int(arg.split("=", 1)[1])
    elif arg.startswith("--steps="):
        CUSTOM_STEPS = int(arg.split("=", 1)[1])
    elif arg.startswith("--resume="):
        RESUME_FROM = arg.split("=", 1)[1]
    elif arg == "--resume":
        RESUME_FROM = "auto"

# Kas baasmudel on lokaalne checkpoint (LoRA juba küljes) või HF mudel?
FROM_CHECKPOINT = Path(BASE_MODEL).exists()
print(f"LoRA rank: {LORA_RANK}")
if not FROM_CHECKPOINT:
    print(f"HF baasmudel – get_peft_model() kutsutakse (rank={LORA_RANK})")

DATA_CSV    = "data/kurrent/metadata.csv"
DATA_IMAGES = "data/kurrent/images"

# Holdout: ~70 lehte, mis jäävad treeningust välja, et mudeleid saaks
# omavahel võrrelda. Nimekirja teeb scripts/make_holdout.py ja see on
# repositooriumis – ilma selleta ei ole hilisem võrdlus aus.
# --no-holdout treenib kõige peal (nt lõplik mudel pärast katsetamist).
HOLDOUT_PATH = "data/kurrent/holdout.txt"
USE_HOLDOUT  = "--no-holdout" not in sys.argv

DATE_STAMP  = datetime.now().strftime("%Y%m%d")
OUTPUT_PATH = f"models/qwen3.5-ocr-kurrent-{DATE_STAMP}"
CKPT_DIR    = f"models/checkpoints-kurrent-{DATE_STAMP}"

# --resume: 33-tunnine jooks ületab südaöö, seega tänane DATE_STAMP oleks juba
# teine kaust. Jätkame selle kaustaga, kust checkpoint tuli.
if RESUME_FROM:
    if RESUME_FROM == "auto":
        dirs = sorted(Path("models").glob("checkpoints-kurrent-*"))
        if not dirs:
            print("Viga: --resume, aga models/checkpoints-kurrent-* puudub.")
            sys.exit(1)
        ckpt_root = dirs[-1]
        ckpts = sorted(ckpt_root.glob("checkpoint-*"),
                       key=lambda x: int(x.name.split("-")[1]))
        if not ckpts:
            print(f"Viga: {ckpt_root} ei sisalda ühtki checkpoint-N kausta.")
            sys.exit(1)
        RESUME_FROM = str(ckpts[-1])
    else:
        if not Path(RESUME_FROM).exists():
            print(f"Viga: checkpoint ei leitud: {RESUME_FROM}")
            sys.exit(1)
        ckpt_root = Path(RESUME_FROM).parent

    CKPT_DIR    = str(ckpt_root)
    DATE_STAMP  = ckpt_root.name.replace("checkpoints-kurrent-", "")
    OUTPUT_PATH = f"models/qwen3.5-ocr-kurrent-{DATE_STAMP}"
    print(f"JÄTKAN checkpointist: {RESUME_FROM}")

print(f"Lähtepunkt:    {BASE_MODEL}")
print(f"Andmestik:     {DATA_CSV}")
print(f"Salvestuskoht: {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Eelkontrollid
# ---------------------------------------------------------------------------

if FROM_CHECKPOINT and not Path(BASE_MODEL).exists():
    print(f"Viga: checkpoint ei leitud: {BASE_MODEL}")
    sys.exit(1)

if not Path(DATA_CSV).exists():
    print(f"Viga: andmestik puudub: {DATA_CSV}")
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

if FROM_CHECKPOINT:
    # Checkpoint sisaldab juba LoRA adaptereid – get_peft_model() EI tohi järgneda
    print("Mudel laaditud (LoRA adapterid juba küljes).")
else:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    print(f"LoRA adapterid lisatud (r={LORA_RANK}).")
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Andmestik
# ---------------------------------------------------------------------------

def _load_holdout(path):
    """Holdout-failist failinimede hulk (ilma kaustata, nagu CSV-s)."""
    holdout = set()
    if not (USE_HOLDOUT and Path(path).exists()):
        return holdout
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            holdout.add(os.path.basename(line.split("\t")[-1]))
    return holdout


class KurrentAndmestik:
    def __init__(self, csv_path, images_dir):
        self.samples = []
        skipped = 0
        holdout = _load_holdout(HOLDOUT_PATH)
        held = 0

        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = row.get("transkriptsioon", "")
                if not isinstance(t, str) or not t.strip():
                    skipped += 1
                    continue
                if os.path.basename(row["failinimi"]) in holdout:
                    held += 1
                    continue
                img_path = os.path.join(images_dir, os.path.basename(row["failinimi"]))
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                self.samples.append({
                    "img": img_path,
                    "txt": t.strip(),
                })

        if skipped:
            print(f"  Hoiatus: {skipped} rida vahele jäetud (puuduv pilt/tekst)")
        if held:
            print(f"  Holdout: {held} lehte treeningust välja ({HOLDOUT_PATH})")
        elif USE_HOLDOUT:
            print(f"  Holdout: nimekirja ei leitud ({HOLDOUT_PATH}) – treenin kõige peal")
        else:
            print("  Holdout: --no-holdout, treenin kõige peal")
        print(f"  Andmestik: {len(self.samples)} näidet")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text":  KURRENT_INSTRUCTION},
                        {"type": "image", "image": s["img"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": s["txt"]}],
                },
            ]
        }


dataset = KurrentAndmestik(DATA_CSV, DATA_IMAGES)

if len(dataset) == 0:
    print("Viga: andmestik on tühi.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Treener
# ---------------------------------------------------------------------------

FastVisionModel.for_training(model)

gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu_stats.name}, mälu: {round(gpu_stats.total_memory / 1024**3, 1)} GB")
print(f"Treeninguandmeid: {len(dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max", max_seq_length=8192),
    args=SFTConfig(
        max_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if TEST_MODE else 8,
        warmup_steps=2 if TEST_MODE else 80,
        max_steps=5 if TEST_MODE else CUSTOM_STEPS,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=1 if TEST_MODE else 10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=CKPT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0,
        save_strategy="no" if TEST_MODE else "steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
    ),
)

class EpohhiLopuAdapter(TrainerCallback):
    """Iga epohhi lõpus adapteri koopia kausta, mida save_total_limit ei rotee.

    Rotatsioon kustutab ainult `checkpoint-N` kaustu, seega `epohh-N-adapter`
    jääb alles. 20260602 jooksul selgus, et teine epohh annab vähe (loss
    0,12 → 0,08) – epohh-1 adapter on hilisemaks A/B-ks vajalik.
    Ainult adapter (~800 MB), ilma optimeerija olekuta: see ei ole
    jätkamispunkt, vaid hindamiseks."""

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if TEST_MODE:
            return
        n = int(round(state.epoch))
        path = f"{CKPT_DIR}/epohh-{n}-adapter"
        (model or trainer.model).save_pretrained(path)
        print(f"\n  [epohh {n}, samm {state.global_step}] adapter salvestatud: {path}")


trainer.add_callback(EpohhiLopuAdapter())

if not TEST_MODE:
    print(f"Checkpoint iga {SAVE_STEPS} sammu järel (~{SAVE_STEPS * 27.9 / 3600:.1f} h), "
          f"alles 3 tk ({3 * 1.2:.1f} GB) → {CKPT_DIR}")
    print(f"Katkemise korral jätkamine: venv/bin/python scripts/train_kurrent.py --resume")

print("Alustan treeningut...")
trainer_stats = trainer.train(resume_from_checkpoint=RESUME_FROM)
print(f"Treening lõppenud! Aeg: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} min")

# ---------------------------------------------------------------------------
# Salvestamine
# ---------------------------------------------------------------------------

if TEST_MODE:
    print("Testrežiim: salvestamine vahele jäetud.")
else:
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\nLoRA adapter salvestatud: {OUTPUT_PATH}")
    print(f"\nSee on eraldiseisev mudel trükiteksti OCR-teenusest.")
    print(f"Kasutamiseks inference'il:")
    print(f"  model, tokenizer = FastVisionModel.from_pretrained('{OUTPUT_PATH}', load_in_4bit=True)")

# ---------------------------------------------------------------------------
# Testinferentsi näide
# ---------------------------------------------------------------------------

print("\nTestan treenitud mudelit...")
FastVisionModel.for_inference(model)

test_images = sorted(
    f for f in Path(DATA_IMAGES).iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
)[:3]

for test_image_path in test_images:
    image = PILImage.open(test_image_path)
    print(f"\nTestpilt: {test_image_path.name}")

    messages = [{"role": "user", "content": [
        {"type": "text",  "text":  KURRENT_INSTRUCTION},
        {"type": "image"},
    ]}]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True, do_sample=False)
    raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    # Eemalda võimalikud thinking tokenid
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    print("--- TULEMUS (esimesed 300 tähemärki) ---")
    print(raw[:300])
