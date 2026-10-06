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
  python scripts/eval_kurrent.py --name kordus-2026xxxx        # muu väljundikaust
  python scripts/eval_kurrent.py --endpoint ... --sampler '{"dry_multiplier":0.8}' nimi
  python scripts/eval_kurrent.py --dry-run                    # kontrolli nimekiri, ei laadi mudelit

llama.cpp GGUF-i mõõtmine käib sama holdout'i ja sama juhisega, aga mudeli
asemel antakse serveri aadress. Server tuleb ise käima panna (teises tmux-i
aknas, ocr-service peatatud):

  ~/Dokumendid/LLM/llama.cpp/build/bin/llama-server \
      -m models/gguf/kurrent-20260602-Q8_0.gguf \
      --mmproj models/gguf/mmproj-kurrent-20260602-F16.gguf \
      --image-max-tokens 5000 \\
      -ngl 99 -c 65536 -np 4 -cb -fa on --host 127.0.0.1 --port 8080

LÕKS 1: **`--image-max-tokens 5000` on kohustuslik.** llama.cpp vaikepiir on
4096 visuaaltokenit (`clip.cpp: set_limit_image_tokens(8, 4096)`), meie eelarve
5 120 000 px / 1024 = 5000. Ilma liputa kärbitakse pilt vaikselt ja väike tekst
kaob. Kontrolli: tekstipäring annab prompt_tokens = juhis+mall, pildiga päring
vahe = visuaaltokenid.

LÕKS 2: `-c` on llama.cpp-s KOKKU kõigi slottide peale, ehk -np 4 puhul saab
iga slot 65536/4 = 16384 tokenit. Leht vajab ~4000 visuaaltokenit + kuni 4096
väljundit, seega -c 32768 (8192/slot) jääks napiks ja server lõikaks konteksti.

  python scripts/eval_kurrent.py --endpoint http://127.0.0.1:8080 kurrent-20260602-Q8_0

Väljund: data/kurrent/eval/<mudeli-nimi>/
  <failinimi>.txt   mudeli väljund
  results.csv       failinimi, allikas, gt_chars, out_chars, ratio, cer, wer

Mõõdikud:
  CER    – tähemärkide veamäär GT vastu (editdistance)
  ratio  – väljundi pikkus / GT pikkus. See püüab kinni kaks viga, mida
           CER üksi ei erista: poole lehe vahelejätmine (~0,5) ja loopi
           jooksmine (>1,5). Vt SPIKKER, "Tühjad ja hõredad leheküljed".

Kaks asja, mis endpoint-režiimis erinevad ja mida ei tohi ära unustada:
  - Pildid skaleeritakse siin ise `imaging.fit_to_budget()`-iga. llama.cpp
    kasutab ametlikku preprocessor-konfi (16M px), meie treening 5,12M px –
    ilma selleta tekiks treening/inferents-nihe (vt image_preprocessing).
  - Serverile antakse `enable_thinking: false`. Ilma selleta tuleb väljundi
    ette tühi <think></think> plokk; `strip_output()` võtab selle küll maha,
    aga tokenid on siis juba genereeritud ja kiirusnumber vale.

NB! CER mõõdab vastavust arhiivikorpuse transkribeerimistavadele, mitte
seda, kui kasulik mudel VUTT-is on. Kasuta seda regressiooni tuvastamiseks
("kas midagi läks katki"), mitte ainsa otsustajana.
"""

import base64
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import editdistance
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from imaging import MAX_PIXELS, fit_to_grid
from prompt import INSTRUCTION, KURRENT_INSTRUCTION
from lyhend_makron import makroniks

csv.field_size_limit(10 ** 7)

DATA_CSV     = Path("data/kurrent/metadata.csv")
DATA_IMAGES  = Path("data/kurrent/images")
HOLDOUT_PATH = Path("data/kurrent/holdout.txt")
EVAL_ROOT    = Path("data/kurrent/eval")

MODEL_PATH = "models/qwen3.5-ocr-kurrent-20260602"
ENDPOINT = None    # llama.cpp server; siis MODEL_PATH on ainult väljundi nimi
OUT_NAME = None    # väljundikausta nimi, kui ei taha mudeli nime järgi
SAMPLER = {}       # lisaparameetrid serveri päringusse (llama.cpp samplerid)
RESAMPLE = "lanczos"   # --resample bicubic jäljendab image processori skaleerimist
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
    elif a == "--name" and i + 1 < len(args):
        OUT_NAME = args[i + 1]; i += 2
    elif a.startswith("--name="):
        OUT_NAME = a.split("=", 1)[1]; i += 1
    elif a == "--resample" and i + 1 < len(args):
        RESAMPLE = args[i + 1]; i += 2
    elif a.startswith("--resample="):
        RESAMPLE = a.split("=", 1)[1]; i += 1
    elif a == "--sampler" and i + 1 < len(args):
        SAMPLER = json.loads(args[i + 1]); i += 2
    elif a.startswith("--sampler="):
        SAMPLER = json.loads(a.split("=", 1)[1]); i += 1
    elif a == "--endpoint" and i + 1 < len(args):
        ENDPOINT = args[i + 1].rstrip("/"); i += 2
    elif a.startswith("--endpoint="):
        ENDPOINT = a.split("=", 1)[1].rstrip("/"); i += 1
    elif a.startswith("--"):
        i += 1                      # --resume jt lipud
    else:
        MODEL_PATH = a; i += 1

if RESAMPLE not in ("lanczos", "bicubic"):
    print("Viga: --resample peab olema 'lanczos' või 'bicubic'")
    sys.exit(1)
RESAMPLE_FILTER = PILImage.LANCZOS if RESAMPLE == "lanczos" else PILImage.BICUBIC

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

if ENDPOINT and MODEL_PATH == "models/qwen3.5-ocr-kurrent-20260602":
    # Vaikimisi mudelitee tähendaks unslothi checkpointi – endpoint-režiimis
    # pole see see, mida mõõdetakse, ja tulemused kirjutaks üle baseline'i.
    print("Viga: --endpoint nõuab väljundi nime, nt")
    print("  python scripts/eval_kurrent.py --endpoint http://127.0.0.1:8080 "
          "kurrent-20260602-Q8_0")
    sys.exit(1)

out_dir = EVAL_ROOT / (OUT_NAME or Path(MODEL_PATH).name)
if PROMPT_KIND == "print" and not OUT_NAME:
    out_dir = EVAL_ROOT / f"{Path(MODEL_PATH).name}--print-prompt"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Mudel:   {MODEL_PATH}")
if ENDPOINT:
    print(f"Server:  {ENDPOINT}  (paralleelseid päringuid: {BATCH})")
    print(f"Skaala:  {RESAMPLE}")
    if SAMPLER:
        print(f"Sampler: {json.dumps(SAMPLER, ensure_ascii=False)}")
print(f"Juhis:   {PROMPT_KIND}")
print(f"Lehti:   {len(holdout)}")
if not ENDPOINT:
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
# Mudel – bf16, nagu teenuseski (4-bit ainult aeglustaks) – või server
# ---------------------------------------------------------------------------

if ENDPOINT is None:
    # torch ja unsloth ainult siin: endpoint-jooks peab käima ka siis, kui
    # GPU on llama-serveri all kinni.
    import torch
    from unsloth import FastVisionModel

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
else:
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=10) as r:
            r.read()
    except Exception as e:
        print(f"Viga: server {ENDPOINT} ei vasta ({e}).")
        print("Käivita llama-server – käsk on selle skripti päises.")
        sys.exit(1)
    print("Server vastab.\n")


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


def pildi_data_uri(path: Path) -> str:
    """Pilt base64 data-URI-ks, TÄPSELT sellel võrel, mida treening nägi.

    Kaks asja, mis on mõõdetud 27.08.2026 ja mida ei tohi tagasi keerata:

    1. **PNG, mitte JPEG.** Lähtefail on juba JPEG; JPEG-ina ümber kodeerides
       tuleb teine põlvkond, mis sööb õhukesed kaldkirjatähed ära. Trüki-
       marginaalid kadusid tervetel lehtedel (`<m>` tage 76 JPEG-iga vs 150
       PNG-ga samal 8 lehel).
    2. **`imaging.fit_to_grid()`**, mitte `fit_to_budget()`. llama.cpp ümardab
       pildi ise 32 kordseks, aga ilma antialiasinguta; kui klient annab pildi
       juba õigel võrel, ei ole tal midagi skaleerida.

    Serverile tuleb lisaks anda **`--image-max-tokens 5000`** (= MAX_PIXELS /
    1024), muidu kehtib llama.cpp vaikepiir 4096 ja pilt kärbitakse vaikselt.
    """
    with PILImage.open(path) as im:
        pilt = fit_to_grid(im.convert("RGB"), resample=RESAMPLE_FILTER)
        buf = io.BytesIO()
        pilt.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saada_serverile(failinimi: str) -> str:
    """Üks leht OpenAI-ühilduva /v1/chat/completions kaudu."""
    keha = json.dumps({
        **SAMPLER,      # nt {"dry_multiplier": 0.8} – llama.cpp laiendus
        "model": "kurrent",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": pildi_data_uri(DATA_IMAGES / Path(failinimi).name)}},
        ]}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        # llama.cpp chat template lisab muidu tühja <think></think> ploki
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions", data=keha,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        vastus = json.loads(r.read())
    return strip_output(vastus["choices"][0]["message"]["content"])


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

if ENDPOINT:
    # llama-server -np N teenindab N päringut korraga; rohkem kliendipoolset
    # paralleelsust sellest kiiremaks ei tee, seega BATCH = -np väärtus.
    def too(kirje):
        allikas, failinimi = kirje
        if not (DATA_IMAGES / Path(failinimi).name).exists():
            print(f"PUUDUB pilt: {failinimi}")
            return allikas, failinimi, None
        try:
            return allikas, failinimi, saada_serverile(failinimi)
        except Exception as e:
            print(f"  VIGA {failinimi}: {e}")
            return allikas, failinimi, None

    with ThreadPoolExecutor(max_workers=BATCH) as pool:
        for allikas, failinimi, hyp in pool.map(too, ootel):
            if hyp is None:
                continue    # faili ei kirjuta -> --resume proovib uuesti
            (out_dir / f"{Path(failinimi).stem}.txt").write_text(hyp, encoding="utf-8")
            rows.append((allikas, failinimi, hyp))
            tehtud += 1
            if tehtud % BATCH == 0 or tehtud == len(ootel):
                kulunud = time.time() - t0
                print(f"[{tehtud}/{len(ootel)}] {kulunud / 60:5.1f} min"
                      f"  ({kulunud / tehtud:4.1f} s/lk)")
else:
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
    # Lühendusmärk tilde ≡ makron (VUTT ADR 0062): GT on alates v4-st makroniga,
    # vanemad mudelid kirjutavad tilde — muidu oleks CER-i erinevus kunstlik.
    # Pikk s ≡ s (06.10): GT on alates v4 pikk-s-ühtlustusest s-iga.
    ref = makroniks(gt[failinimi].strip())[0].replace("ſ", "s")
    hyp = makroniks(hyp)[0].replace("ſ", "s")
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
kulunud = time.time() - t0
print(f"Aeg: {kulunud / 60:.1f} min"
      + (f"  ({kulunud / tehtud:.1f} s/lk, {3600 * tehtud / kulunud:.0f} lehte/tunnis)"
         if tehtud else ""))

# Kiirus on siin sama tähtis mõõdik kui CER (kas mootori vahetus tasub ära),
# aga see elab muidu ainult terminali ajaloos. Paneme jooksu tingimused
# kettale, muidu pole hilisem võrdlus aus: kell ei mõõda ainult mudelit, vaid
# ka GPU voolupiiri ja seda, mitu lehte tegelikult joosti.
run_json = out_dir / "run.json"
tingimused = {
    "mudel": MODEL_PATH,
    "endpoint": ENDPOINT,
    "juhis": PROMPT_KIND,
    "batch": BATCH,                     # endpoint-režiimis paralleelsed päringud
    "max_new_tokens": MAX_NEW_TOKENS,
    "pildieelarve_px": MAX_PIXELS,
    "sampler": SAMPLER or None,
    "resample": RESAMPLE,
    "lehti_kokku": len(rows),
    "lehti_seekord": tehtud,            # resume'i puhul vähem kui kokku
    "sekundeid": round(kulunud, 1),
    "s_per_lk": round(kulunud / tehtud, 2) if tehtud else None,
    "lehte_tunnis": round(3600 * tehtud / kulunud, 1) if tehtud else None,
    "aeg": time.strftime("%Y-%m-%d %H:%M"),
}
try:
    import subprocess
    tingimused["gpu_power_limit_W"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()[0].strip()
except Exception:
    pass
run_json.write_text(json.dumps(tingimused, ensure_ascii=False, indent=2), encoding="utf-8")

if tehtud < len(rows):
    print("NB! Osa lehti tuli --resume abil kettalt – ajanumber ei kata kogu jooksu.")
print(f"Tulemused: {results_csv}")
