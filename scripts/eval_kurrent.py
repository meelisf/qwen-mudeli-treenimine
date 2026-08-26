#!/usr/bin/env python3
"""
Kurrent-mudeli hindamine holdout-lehtede peal

Laseb mudeli üle nende ~70 lehe, mis `make_holdout.py` treeningust välja
jättis, ja võrdleb väljundit andmestiku ground truth'iga. Väljundid jäävad
kettale, nii et hiljem saab neid ka silmaga kõrvuti vaadata.

  python scripts/eval_kurrent.py                              # vana mudel
  python scripts/eval_kurrent.py models/qwen3.5-ocr-kurrent-20260828
  python scripts/eval_kurrent.py --prompt print               # INSTRUCTION, mitte KURRENT_INSTRUCTION
  python scripts/eval_kurrent.py --limit 10                   # kiire proov
  python scripts/eval_kurrent.py --resume                     # jätka pooleli jäänut
  python scripts/eval_kurrent.py --dry-run                    # kontrolli nimekiri, ei laadi mudelit

Väljund: data/kurrent/eval/<mudeli-nimi>/
  <failinimi>.txt   mudeli väljund
  results.csv       failinimi, allikas, gt_chars, out_chars, ratio, cer, wer

Mõõdikud:
  CER    – tähemärkide veamäär GT vastu (editdistance)
  ratio  – väljundi pikkus / GT pikkus. See püüab kinni kaks viga, mida
           CER üksi ei erista: poole lehe vahelejätmine (~0,5) ja loopi
           jooksmine (>1,5). Vt SPIKKER, "Tühjad ja hõredad leheküljed".

NB! CER mõõdab vastavust arhiivikorpuse transkribeerimistavadele, mitte
seda, kui kasulik mudel VUTT-is on. Kasuta seda regressiooni tuvastamiseks
("kas midagi läks katki"), mitte ainsa otsustajana.
"""

import csv
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import editdistance
import torch
from PIL import Image as PILImage
from unsloth import FastVisionModel

sys.path.insert(0, str(Path(__file__).parent))
from imaging import MAX_PIXELS
from prompt import INSTRUCTION, KURRENT_INSTRUCTION

csv.field_size_limit(10 ** 7)

DATA_CSV     = Path("data/kurrent/metadata.csv")
DATA_IMAGES  = Path("data/kurrent/images")
HOLDOUT_PATH = Path("data/kurrent/holdout.txt")
EVAL_ROOT    = Path("data/kurrent/eval")

MODEL_PATH = "models/qwen3.5-ocr-kurrent-20260602"
PROMPT_KIND = "kurrent"
LIMIT = None
MAX_NEW_TOKENS = 4096
BATCH = 4          # sama mis teenuses; tipp ~24,6 GB / 32,6 GB
RESUME = "--resume" in sys.argv
DRY_RUN = "--dry-run" in sys.argv   # kontrolli nimekiri üle, mudelit ei laeta

args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--prompt" and i + 1 < len(args):
        PROMPT_KIND = args[i + 1]; i += 2
    elif a.startswith("--prompt="):
        PROMPT_KIND = a.split("=", 1)[1]; i += 1
    elif a == "--limit" and i + 1 < len(args):
        LIMIT = int(args[i + 1]); i += 2
    elif a.startswith("--limit="):
        LIMIT = int(a.split("=", 1)[1]); i += 1
    elif a == "--max-new-tokens" and i + 1 < len(args):
        MAX_NEW_TOKENS = int(args[i + 1]); i += 2
    elif a.startswith("--max-new-tokens="):
        MAX_NEW_TOKENS = int(a.split("=", 1)[1]); i += 1
    elif a == "--batch" and i + 1 < len(args):
        BATCH = int(args[i + 1]); i += 2
    elif a.startswith("--batch="):
        BATCH = int(a.split("=", 1)[1]); i += 1
    elif a.startswith("--"):
        i += 1                      # --resume jt lipud
    else:
        MODEL_PATH = a; i += 1

if PROMPT_KIND not in ("kurrent", "print"):
    print("Viga: --prompt peab olema 'kurrent' või 'print'")
    sys.exit(1)

# Teenus (kataloogi-jalgimine-ja-ocr.py) saadab käsikirjamudelile praegu
# INSTRUCTION-i, kuigi treenitud on KURRENT_INSTRUCTION-iga. --prompt print
# on selleks, et selle vahe saaks lõpuks ära mõõta.
PROMPT = KURRENT_INSTRUCTION if PROMPT_KIND == "kurrent" else INSTRUCTION

if not HOLDOUT_PATH.exists():
    print(f"Viga: holdout puudub: {HOLDOUT_PATH}")
    print("  Tee see kõigepealt: python scripts/make_holdout.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Holdout + ground truth
# ---------------------------------------------------------------------------

holdout = []          # (allikas, failinimi)
for line in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        allikas, failinimi = line.split("\t")
        holdout.append((allikas, failinimi))

gt = {}
with open(DATA_CSV, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        gt[row[0]] = row[1]

missing = [f for _, f in holdout if f not in gt]
if missing:
    print(f"Viga: {len(missing)} holdout-lehte pole andmestikus, nt {missing[0]}")
    sys.exit(1)

if LIMIT:
    holdout = holdout[:LIMIT]

out_dir = EVAL_ROOT / Path(MODEL_PATH).name
if PROMPT_KIND == "print":
    out_dir = EVAL_ROOT / f"{Path(MODEL_PATH).name}--print-prompt"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Mudel:   {MODEL_PATH}")
print(f"Juhis:   {PROMPT_KIND}")
print(f"Lehti:   {len(holdout)}")
print(f"Batch:   {BATCH}")
print(f"Väljund: {out_dir}")

if DRY_RUN:
    from collections import Counter
    c = Counter(a for a, _ in holdout)
    print(f"\n{'allikas':26s} {'lk':>3s} {'GT märke keskm.':>16s}")
    for allikas in sorted(c):
        pikkused = [len(gt[f].strip()) for a, f in holdout if a == allikas]
        print(f"{allikas:26s} {c[allikas]:3d} {sum(pikkused) // len(pikkused):16d}")
    puuduvad = [f for _, f in holdout if not (DATA_IMAGES / Path(f).name).exists()]
    print(f"\nPuuduvaid pilte: {len(puuduvad)}")
    print("--dry-run: mudelit ei laetud.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Mudel – bf16, nagu teenuseski (4-bit ainult aeglustaks)
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
tokenizer.image_processor.size = {
    "longest_edge": MAX_PIXELS,
    "shortest_edge": tokenizer.image_processor.size.get("shortest_edge", 65536),
}
FastVisionModel.for_inference(model)
print("Mudel laaditud.\n")


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return editdistance.eval(r, h) / len(r)


import re


def strip_output(text: str) -> str:
    """Sama puhastus mis teenuses: think-plokid, assistendi markerid, koodiplokid.

    Batchis on promptid eri pikkusega (pildi tokenite arv erineb), seega
    lõikame prompti ära markeri järgi, mitte indeksi järgi.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for marker in ["</assistant>", "<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def genereeri(images):
    """Üks batch pilte -> list puhastatud väljundeid."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text",  "text": PROMPT},
            {"type": "image"},
        ]}],
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(
        images, [chat] * len(images),
        add_special_tokens=False, return_tensors="pt", padding=True,
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()
    return [strip_output(t) for t in decoded]


rows = []
t0 = time.time()
tehtud = 0

# Resume: valmis lehed loeme kettalt, ülejäänud lähevad batchidesse
ootel = []
for allikas, failinimi in holdout:
    out_path = out_dir / f"{Path(failinimi).stem}.txt"
    if RESUME and out_path.exists():
        rows.append((allikas, failinimi, out_path.read_text(encoding="utf-8").strip()))
    else:
        ootel.append((allikas, failinimi))

if rows:
    print(f"Resume: {len(rows)} lehte juba olemas, jäänud {len(ootel)}\n")

for algus in range(0, len(ootel), BATCH):
    grupp = ootel[algus:algus + BATCH]
    images, kehtivad = [], []
    for allikas, failinimi in grupp:
        img_path = DATA_IMAGES / Path(failinimi).name
        if not img_path.exists():
            print(f"PUUDUB pilt: {img_path}")
            continue
        images.append(PILImage.open(img_path).convert("RGB"))
        kehtivad.append((allikas, failinimi))
    if not images:
        continue

    try:
        valjundid = genereeri(images)
    except torch.cuda.OutOfMemoryError:
        # Batch ei mahtunud – proovi ükshaaval, et jooks ei kukuks
        print("  OOM – proovin ükshaaval")
        torch.cuda.empty_cache()
        valjundid = []
        for img in images:
            valjundid.extend(genereeri([img]))

    for (allikas, failinimi), hyp in zip(kehtivad, valjundid):
        (out_dir / f"{Path(failinimi).stem}.txt").write_text(hyp, encoding="utf-8")
        rows.append((allikas, failinimi, hyp))
        tehtud += 1

    for img in images:
        img.close()

    kulunud = time.time() - t0
    print(f"[{algus + len(kehtivad)}/{len(ootel)}] {kulunud / 60:5.1f} min"
          f"  ({kulunud / max(tehtud, 1):4.1f} s/lk)")

# Mõõdikud
tulemused = []
for allikas, failinimi, hyp in rows:
    ref = gt[failinimi].strip()
    ratio = len(hyp) / max(len(ref), 1)
    tulemused.append({
        "failinimi": failinimi, "allikas": allikas,
        "gt_chars": len(ref), "out_chars": len(hyp),
        "ratio": round(ratio, 3),
        "cer": round(cer(ref, hyp), 4), "wer": round(wer(ref, hyp), 4),
    })
rows = tulemused

# ---------------------------------------------------------------------------
# Kokkuvõte
# ---------------------------------------------------------------------------

results_csv = out_dir / "results.csv"
with open(results_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def keskmine(values):
    return sum(values) / len(values) if values else float("nan")


print(f"\n{'allikas':26s} {'lk':>3s} {'CER':>7s} {'WER':>7s} {'ratio':>7s}")
for allikas in sorted({r["allikas"] for r in rows}):
    grupp = [r for r in rows if r["allikas"] == allikas]
    print(f"{allikas:26s} {len(grupp):3d} "
          f"{keskmine([r['cer'] for r in grupp]):6.1%} "
          f"{keskmine([r['wer'] for r in grupp]):6.1%} "
          f"{keskmine([r['ratio'] for r in grupp]):7.2f}")
print(f"{'KOKKU':26s} {len(rows):3d} "
      f"{keskmine([r['cer'] for r in rows]):6.1%} "
      f"{keskmine([r['wer'] for r in rows]):6.1%} "
      f"{keskmine([r['ratio'] for r in rows]):7.2f}")

luhikesed = [r for r in rows if r["ratio"] < 0.7]
pikad     = [r for r in rows if r["ratio"] > 1.4]
print(f"\nLühikesi (<0,7): {len(luhikesed)}   pikki (>1,4): {len(pikad)}")
print(f"Aeg: {(time.time() - t0) / 60:.1f} min")
print(f"Tulemused: {results_csv}")
