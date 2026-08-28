#!/usr/bin/env python3
"""
Trükimudeli hindamine VUTT holdout-lehtede peal

Vaste `eval_kurrent.py`-le trükipoolel. Laseb mudeli üle nende 20 lehe, mille
`make_holdout_print.py` treeningust välja jättis, ja võrdleb väljundit
andmestiku ground truth'iga.

  python scripts/eval_print.py models/qwen3.5-ocr-print-base-r64-20260827
  python scripts/eval_print.py models/qwen3.5-ocr-markup-20260720   # baseline
  python scripts/eval_print.py --limit 4 --name proov <mudel>
  python scripts/eval_print.py --resume <mudel>
  python scripts/eval_print.py --dry-run

llama.cpp GGUF mõõdetakse sama holdout'i ja sama juhisega, ainult mudeli
asemel antakse serveri aadress (server käivita ise teises tmux-i aknas):

  ~/Dokumendid/LLM/llama.cpp/build/bin/llama-server \
      -m models/gguf/print-base-r64-20260827-Q8_0.gguf \
      --mmproj models/gguf/mmproj-print-base-r64-20260827-F16.gguf \
      --image-max-tokens 5000 \
      -ngl 99 -c 65536 -np 4 -cb -fa on --host 127.0.0.1 --port 8080

  python scripts/eval_print.py --endpoint http://127.0.0.1:8080 print-base-r64-Q8_0

LÕKS 1: **`--image-max-tokens 5000` on kohustuslik.** llama.cpp vaikepiir on
4096 visuaaltokenit (`clip.cpp: set_limit_image_tokens(8, 4096)`), meie eelarve
5 120 000 px / 1024 = 5000. Ilma liputa kärbitakse pilt vaikselt ~8 % ja kitsas
ääretekst kaob – täpselt see, mida `m_*` veerud siin mõõdavad.

LÕKS 2: klient saadab **PNG**, mitte JPEG. Lähtefail on juba JPEG; teine
JPEG-põlvkond sööb õhukesed kaldkirjatähed ära (`<m>` 76 vs 150 samal 8 lehel).

LÕKS 3: `-c` on llama.cpp-s KOKKU kõigi slottide peale, ehk `-np 4` puhul saab
iga slot 65536/4 = 16384 tokenit. Leht vajab ~5000 visuaaltokenit + kuni 4096
väljundit, seega `-c 32768` jääks napiks.

**Miks eraldi skript, mitte `--corpus` lipp eval_kurrent'is.** Trükipoolel ei
ole CER peamine mõõdik. Plaani punkt 0b: `<m>` on ainus sine qua non. Seega
mõõdame kolme telge korraga ja hoiame nad eraldi veergudes:

  cer / wer / ratio   kogu märgendatud tekst, nagu treening seda nägi
  cer_plain           märgendid maha võetud – puhas transkriptsioonikvaliteet
  m_*                 marginaalid: mitu, kui täpselt, kus puudu / juurde

Ilma `cer_plain`-ita ei saa eristada kahte täiesti erinevat viga: mudel loeb
teksti valesti vs mudel loeb teksti õigesti, aga paneb märgendid valesse
kohta. Esimene on OCR-i regressioon, teine on märgendusregressioon.

GROUND TRUTH käib läbi `clean_markup()` – TÄPSELT sama ahel mis
`train_markup.py`-s (rida 195). Ilma selleta mõõdaks CER seda, kui hästi
mudel jäljendab puhastamata VUTT-i, mida talle kunagi ei näidatud.

NB! 20 lehte **ankurdavad** kvaliteeti ja püüavad regressiooni. Korpuse CER-i
nad ei anna – vt plaani punkt 4, kiht 1.

Väljund: data/vutt/eval/<mudeli-nimi>/
  <failinimi>.txt   mudeli väljund
  results.csv       lehekülje kaupa kõik mõõdikud
  run.json          jooksu tingimused (mudel, kiirus, voolupiir)
"""

import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import editdistance
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from convert_marginalia import clean_markup
from imaging import MAX_PIXELS, fit_to_grid
from prompt import INSTRUCTION

csv.field_size_limit(10 ** 7)

DATA_CSV     = Path("data/vutt/metadata.csv")
DATA_IMAGES  = Path("data/vutt/images")
HOLDOUT_PATH = Path("data/vutt/holdout.txt")
EVAL_ROOT    = Path("data/vutt/eval")

MODEL_PATH = None
ENDPOINT   = None    # llama.cpp server; siis MODEL_PATH on ainult väljundi nimi
OUT_NAME   = None
LIMIT      = None
MAX_NEW_TOKENS = 4096
BATCH   = 4          # sama mis teenuses; tipp ~24,6 GB / 32,6 GB
RESUME  = "--resume" in sys.argv
#: Peab vastama sellele, millega mudel treeniti – muidu mõõdab CER vale GT
#: vastu. `<m>`-mõõdikuid see ei mõjuta (`normaliseeri()` võtab tagid maha).
KEEP_M_ITALICS = "--keep-m-italics" in sys.argv
DRY_RUN = "--dry-run" in sys.argv

args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--limit" and i + 1 < len(args):
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
    elif a == "--endpoint" and i + 1 < len(args):
        ENDPOINT = args[i + 1].rstrip("/"); i += 2
    elif a.startswith("--endpoint="):
        ENDPOINT = a.split("=", 1)[1].rstrip("/"); i += 1
    elif a == "--name" and i + 1 < len(args):
        OUT_NAME = args[i + 1]; i += 2
    elif a.startswith("--name="):
        OUT_NAME = a.split("=", 1)[1]; i += 1
    elif a.startswith("--"):
        i += 1
    else:
        MODEL_PATH = a; i += 1

if MODEL_PATH is None and not DRY_RUN:
    print("Kasutus: python scripts/eval_print.py <mudeli-tee>")
    print("  nt models/qwen3.5-ocr-print-base-r64-20260827")
    print("  või: --endpoint http://127.0.0.1:8080 <väljundi-nimi>")
    sys.exit(1)

if not HOLDOUT_PATH.exists():
    print(f"Viga: holdout puudub: {HOLDOUT_PATH}")
    print("  Tee see kõigepealt: python scripts/make_holdout_print.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Holdout + ground truth
# ---------------------------------------------------------------------------

# Erinevalt Kurrendist on siin üks rida = üks failinimi (ei ole allikaveergu).
# Kiht tuletatakse GT `<m>` arvust, sama reegliga mis make_holdout_print.py-s.
holdout = [
    rida.strip() for rida in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
    if rida.strip() and not rida.startswith("#")
]

gt = {}
with open(DATA_CSV, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = row.get("transkriptsioon") or ""
        if t.strip():
            # basename, sest metadata.csv-s on "images/xxx.jpg", holdout'is mitte
            gt[os.path.basename(row["failinimi"])] = clean_markup(
                t, keep_marginalia_italics=KEEP_M_ITALICS).strip()

missing = [f for f in holdout if f not in gt]
if missing:
    print(f"Viga: {len(missing)} holdout-lehte pole andmestikus, nt {missing[0]}")
    sys.exit(1)

if LIMIT:
    holdout = holdout[:LIMIT]

M_TAG = re.compile(r"<m>(.*?)</m>", re.DOTALL)
TAG   = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\s*/?>")
ROHKE_PIIR = 4       # sama piir mis make_holdout_print.py-s


def m_sisud(text: str) -> list:
    return [s.strip() for s in M_TAG.findall(text) if s.strip()]


def kiht(ref: str) -> str:
    n = len(m_sisud(ref))
    if n >= ROHKE_PIIR:
        return "marginaalirohke"
    return "marginaaliga" if n else "ilma"


out_dir = EVAL_ROOT / (OUT_NAME or Path(MODEL_PATH or "dry-run").name)

print(f"Mudel:   {MODEL_PATH}")
if ENDPOINT:
    print(f"Server:  {ENDPOINT}  (paralleelseid päringuid: {BATCH})")
print(f"Juhis:   print (INSTRUCTION)")
print(f"GT:      <i> <m> sees {'ALLES' if KEEP_M_ITALICS else 'eemaldatud'}")
print(f"Lehti:   {len(holdout)}")
print(f"Batch:   {BATCH}")
print(f"Väljund: {out_dir}")

if DRY_RUN:
    from collections import Counter
    c = Counter(kiht(gt[f]) for f in holdout)
    print(f"\n{'kiht':18s} {'lk':>3s} {'<m>':>5s} {'GT märke keskm.':>16s}")
    for k in ("marginaalirohke", "marginaaliga", "ilma"):
        grupp = [f for f in holdout if kiht(gt[f]) == k]
        if not grupp:
            continue
        print(f"{k:18s} {c[k]:3d} {sum(len(m_sisud(gt[f])) for f in grupp):5d} "
              f"{sum(len(gt[f]) for f in grupp) // len(grupp):16d}")
    puuduvad = [f for f in holdout if not (DATA_IMAGES / f).exists()]
    print(f"\nPuuduvaid pilte: {len(puuduvad)}")
    print("--dry-run: mudelit ei laetud.")
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Mõõdikud
# ---------------------------------------------------------------------------


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


def strip_tags(text: str) -> str:
    return re.sub(r"[ \t]+", " ", TAG.sub("", text)).strip()


def normaliseeri(s: str) -> str:
    """Marginaali sisu võrdlemiseks: märgendid, kirjavahemärgid ja tühik maha.

    Marginaal on tavaliselt lühike viide (`Apoc. 12.`); punkt või tühik
    erinevuse taga ei ole see, mida me mõõta tahame – mõõdame, kas mudel
    NÄGI marginaali ja luges selle õigesti.
    """
    s = strip_tags(s).lower()
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE)


def marginaalid(ref: str, hyp: str) -> dict:
    """Marginaalimõõdikud: kogus, täpne kattuvus, sisu veamäär.

    `m_f1` on hulgapõhine (multiset) täpne kattuvus normaliseeritud sisu peal –
    range, aga tõlgendatav: „mitu marginaali sai mudel sõna-sõnalt kätte".
    `m_cer` võtab kõigi marginaalide sisu järjekorras kokku ja mõõdab CER-i –
    see püüab kinni ka poolikult õige lugemise, mida m_f1 ei näe.
    """
    r_list, h_list = m_sisud(ref), m_sisud(hyp)
    r_norm = [normaliseeri(s) for s in r_list]
    h_norm = [normaliseeri(s) for s in h_list]

    jaanud = list(h_norm)
    tabamusi = 0
    for s in r_norm:
        if s in jaanud:
            jaanud.remove(s)
            tabamusi += 1

    prec   = tabamusi / len(h_norm) if h_norm else (1.0 if not r_norm else 0.0)
    recall = tabamusi / len(r_norm) if r_norm else (1.0 if not h_norm else 0.0)
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    return {
        "m_gt": len(r_list), "m_out": len(h_list), "m_hit": tabamusi,
        "m_prec": round(prec, 3), "m_recall": round(recall, 3),
        "m_f1": round(f1, 3),
        "m_cer": round(cer(" ".join(r_norm), " ".join(h_norm)), 4),
    }


def strip_output(text: str) -> str:
    """Sama puhastus mis teenuses: think-plokid, assistendi markerid, koodiplokid."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for marker in ["</assistant>", "<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Mudel – bf16, nagu teenuseski (4-bit ainult aeglustaks) – või llama-server
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


def pildi_data_uri(path: Path) -> str:
    """Pilt base64 data-URI-ks, TÄPSELT sellel võrel, mida treening nägi.

    PNG (mitte JPEG – teine põlvkond sööb õhukesed kaldkirjatähed) ja
    `fit_to_grid()` (32 kordne võre, et llama.cpp ei peaks ise ilma
    antialiasinguta skaleerima). Vt skripti päise LÕKS 2.
    """
    with PILImage.open(path) as im:
        pilt = fit_to_grid(im.convert("RGB"), resample=PILImage.LANCZOS)
        buf = io.BytesIO()
        pilt.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saada_serverile(failinimi: str) -> str:
    """Üks leht OpenAI-ühilduva /v1/chat/completions kaudu."""
    keha = json.dumps({
        "model": "print",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image_url",
             "image_url": {"url": pildi_data_uri(DATA_IMAGES / failinimi)}},
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
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
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

ootel = []
for failinimi in holdout:
    out_path = out_dir / f"{Path(failinimi).stem}.txt"
    if RESUME and out_path.exists():
        rows.append((failinimi, out_path.read_text(encoding="utf-8").strip()))
    else:
        ootel.append(failinimi)

if rows:
    print(f"Resume: {len(rows)} lehte juba olemas, jäänud {len(ootel)}\n")

if ENDPOINT:
    # llama-server -np N teenindab N päringut korraga; rohkem kliendipoolset
    # paralleelsust sellest kiiremaks ei tee, seega BATCH = -np väärtus.
    def too(failinimi):
        if not (DATA_IMAGES / failinimi).exists():
            print(f"PUUDUB pilt: {failinimi}")
            return failinimi, None
        try:
            return failinimi, saada_serverile(failinimi)
        except Exception as e:
            print(f"  VIGA {failinimi}: {e}")
            return failinimi, None

    with ThreadPoolExecutor(max_workers=BATCH) as pool:
        for failinimi, hyp in pool.map(too, ootel):
            if hyp is None:
                continue    # faili ei kirjuta -> --resume proovib uuesti
            (out_dir / f"{Path(failinimi).stem}.txt").write_text(hyp, encoding="utf-8")
            rows.append((failinimi, hyp))
            tehtud += 1
            if tehtud % BATCH == 0 or tehtud == len(ootel):
                kulunud = time.time() - t0
                print(f"[{tehtud}/{len(ootel)}] {kulunud / 60:5.1f} min"
                      f"  ({kulunud / tehtud:4.1f} s/lk)")
else:
  for algus in range(0, len(ootel), BATCH):
    grupp = ootel[algus:algus + BATCH]
    images, kehtivad = [], []
    for failinimi in grupp:
        img_path = DATA_IMAGES / failinimi
        if not img_path.exists():
            print(f"PUUDUB pilt: {img_path}")
            continue
        images.append(PILImage.open(img_path).convert("RGB"))
        kehtivad.append(failinimi)
    if not images:
        continue

    try:
        valjundid = genereeri(images)
    except torch.cuda.OutOfMemoryError:
        print("  OOM – proovin ükshaaval")
        torch.cuda.empty_cache()
        valjundid = []
        for img in images:
            valjundid.extend(genereeri([img]))

    for failinimi, hyp in zip(kehtivad, valjundid):
        (out_dir / f"{Path(failinimi).stem}.txt").write_text(hyp, encoding="utf-8")
        rows.append((failinimi, hyp))
        tehtud += 1

    for img in images:
        img.close()

    kulunud = time.time() - t0
    print(f"[{algus + len(kehtivad)}/{len(ootel)}] {kulunud / 60:5.1f} min"
          f"  ({kulunud / max(tehtud, 1):4.1f} s/lk)")

tulemused = []
for failinimi, hyp in rows:
    ref = gt[failinimi]
    kirje = {
        "failinimi": failinimi,
        "kiht": kiht(ref),
        "gt_chars": len(ref), "out_chars": len(hyp),
        "ratio": round(len(hyp) / max(len(ref), 1), 3),
        "cer": round(cer(ref, hyp), 4),
        "wer": round(wer(ref, hyp), 4),
        "cer_plain": round(cer(strip_tags(ref), strip_tags(hyp)), 4),
        "i_gt": ref.count("<i>"), "i_out": hyp.count("<i>"),
        "cs_gt": ref.count("<cs>"), "cs_out": hyp.count("<cs>"),
    }
    kirje.update(marginaalid(ref, hyp))
    tulemused.append(kirje)
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


print(f"\n{'kiht':18s} {'lk':>3s} {'CER':>7s} {'CERtxt':>7s} {'ratio':>6s} "
      f"{'<m>gt':>6s} {'<m>out':>7s} {'m_F1':>6s} {'m_CER':>7s}")
for k in ("marginaalirohke", "marginaaliga", "ilma"):
    grupp = [r for r in rows if r["kiht"] == k]
    if not grupp:
        continue
    print(f"{k:18s} {len(grupp):3d} "
          f"{keskmine([r['cer'] for r in grupp]):6.1%} "
          f"{keskmine([r['cer_plain'] for r in grupp]):6.1%} "
          f"{keskmine([r['ratio'] for r in grupp]):6.2f} "
          f"{sum(r['m_gt'] for r in grupp):6d} {sum(r['m_out'] for r in grupp):7d} "
          f"{keskmine([r['m_f1'] for r in grupp]):6.2f} "
          f"{keskmine([r['m_cer'] for r in grupp]):6.1%}")
print(f"{'KOKKU':18s} {len(rows):3d} "
      f"{keskmine([r['cer'] for r in rows]):6.1%} "
      f"{keskmine([r['cer_plain'] for r in rows]):6.1%} "
      f"{keskmine([r['ratio'] for r in rows]):6.2f} "
      f"{sum(r['m_gt'] for r in rows):6d} {sum(r['m_out'] for r in rows):7d} "
      f"{keskmine([r['m_f1'] for r in rows]):6.2f} "
      f"{keskmine([r['m_cer'] for r in rows]):6.1%}")

mediaan = sorted(r["cer"] for r in rows)[len(rows) // 2]
print(f"\nCER mediaan: {mediaan:.1%}")
luhikesed = [r for r in rows if r["ratio"] < 0.7]
pikad     = [r for r in rows if r["ratio"] > 1.4]
print(f"Lühikesi (<0,7): {len(luhikesed)}   pikki (>1,4): {len(pikad)}")
for r in luhikesed + pikad:
    print(f"  {r['ratio']:5.2f}  {r['failinimi']}")

kulunud = time.time() - t0
print(f"Aeg: {kulunud / 60:.1f} min"
      + (f"  ({kulunud / tehtud:.1f} s/lk, {3600 * tehtud / kulunud:.0f} lehte/tunnis)"
         if tehtud else ""))

run_json = out_dir / "run.json"
tingimused = {
    "mudel": MODEL_PATH,
    "endpoint": ENDPOINT,
    "juhis": "print",
    "batch": BATCH,
    "max_new_tokens": MAX_NEW_TOKENS,
    "pildieelarve_px": MAX_PIXELS,
    "lehti_kokku": len(rows),
    "lehti_seekord": tehtud,
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
