#!/usr/bin/env python3
"""
VUTT "Toores" lehtede re-OCR ja võrdlus salvestatud tekstiga

Trükipoolel ei ole ground truth'i ([[test-ground-truth-incomplete]]), aga on
midagi peaaegu sama kasulikku: VUTT-i backup-snapshotis on lehti, mille
praegune tootmismudel on juba transkribeerinud ja mida ükski inimene pole
puutunud. Need on tasuta A/B alus – uus ahel jookseb sama pildi peal, vana
väljund on kettal olemas.

  # server vajab --image-max-tokens 5000 (vt eval_kurrent.py päist)
  python scripts/reocr_vutt.py --endpoint http://127.0.0.1:8080 markup-Q8_0
  python scripts/reocr_vutt.py --endpoint ... --limit 10 --resume proov
  python scripts/reocr_vutt.py --stats          # ainult kandidaatide nimekiri

Valik (--since, --type):
  status == "Toores"           inimene pole teksti kinnitanud
  updated_at >= --since        vaikimisi 2026-07-22, mil markup-20260722
                               aktiveeriti – varasemad on teise mudeli tehtud
  history-s pole "text_edit"   inimene pole teksti käsitsi muutnud

Väljund: data/vutt/reocr/<nimi>/
  <leht>.txt      uus väljund
  results.csv     leht, teos, vana_chars, uus_chars, ratio, erinevus, loop,
                  märgendite arvud vana/uus

NB! CER-i siin EI OLE – vana tekst ei ole tõde, vaid teine arvamus. Numbrid
ütlevad, KUS ahelad lahknevad; kumb on parem, otsustab silm.
"""

import base64
import csv
import io
import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import editdistance
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from build_vutt_dataset import read_work_type
from imaging import MAX_PIXELS, fit_to_grid
from loop_detect import is_looped
from prompt import INSTRUCTION, KURRENT_INSTRUCTION

VUTT_ROOT = Path.home() / "vutt-backups/latest/data"
OUT_ROOT  = Path("data/vutt/reocr")

ENDPOINT = None
OUT_NAME = None
SINCE    = "2026-07-22"       # markup-20260722 aktiveerimise kuupäev
MATERIAL = "print"
LIMIT    = None
BATCH    = 4
MAX_NEW_TOKENS = 4096
RESUME   = "--resume" in sys.argv
STATS    = "--stats" in sys.argv

args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--endpoint" and i + 1 < len(args):
        ENDPOINT = args[i + 1].rstrip("/"); i += 2
    elif a.startswith("--endpoint="):
        ENDPOINT = a.split("=", 1)[1].rstrip("/"); i += 1
    elif a == "--since" and i + 1 < len(args):
        SINCE = args[i + 1]; i += 2
    elif a.startswith("--since="):
        SINCE = a.split("=", 1)[1]; i += 1
    elif a == "--type" and i + 1 < len(args):
        MATERIAL = args[i + 1]; i += 2
    elif a.startswith("--type="):
        MATERIAL = a.split("=", 1)[1]; i += 1
    elif a == "--limit" and i + 1 < len(args):
        LIMIT = int(args[i + 1]); i += 2
    elif a.startswith("--limit="):
        LIMIT = int(a.split("=", 1)[1]); i += 1
    elif a == "--batch" and i + 1 < len(args):
        BATCH = int(args[i + 1]); i += 2
    elif a.startswith("--batch="):
        BATCH = int(a.split("=", 1)[1]); i += 1
    elif a.startswith("--"):
        i += 1
    else:
        OUT_NAME = a; i += 1

PROMPT = INSTRUCTION if MATERIAL == "print" else KURRENT_INSTRUCTION

# ---------------------------------------------------------------------------
# Kandidaadid
# ---------------------------------------------------------------------------

def leia_kandidaadid():
    """Lehed, mille tekst on mudeli tehtud ja inimene pole seda puutunud."""
    valik = []
    for j in VUTT_ROOT.rglob("*.json"):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("status") != "Toores":
            continue
        ua = d.get("updated_at")
        if not ua or ua < SINCE:
            continue
        if any(h.get("action") == "text_edit"
               for h in (d.get("history") or []) if isinstance(h, dict)):
            continue
        if read_work_type(j.parent) != MATERIAL:
            continue
        txt, img = j.with_suffix(".txt"), j.with_suffix(".jpg")
        if not txt.exists() or not img.exists() or txt.stat().st_size < 200:
            continue
        valik.append((ua, j.parent.name, j.stem, img, txt))
    valik.sort(reverse=True)
    return valik


kandidaadid = leia_kandidaadid()
print(f"Allikas: {VUTT_ROOT}")
print(f"Valik:   status=Toores, updated_at >= {SINCE}, type={MATERIAL}, inimene pole muutnud")
print(f"Lehti:   {len(kandidaadid)}  ({len({t for _, t, _, _, _ in kandidaadid})} teost)")

if STATS:
    c = Counter(t for _, t, _, _, _ in kandidaadid)
    for teos, n in c.most_common():
        print(f"  {n:3d}  {teos[:70]}")
    sys.exit(0)

if not ENDPOINT:
    print("\nViga: --endpoint on kohustuslik (llama-server aadress).")
    sys.exit(1)
if not OUT_NAME:
    print("\nViga: anna väljundi nimi, nt: reocr_vutt.py --endpoint ... markup-Q8_0")
    sys.exit(1)

if LIMIT:
    kandidaadid = kandidaadid[:LIMIT]

out_dir = OUT_ROOT / OUT_NAME
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Server:  {ENDPOINT}  (paralleelseid: {BATCH})")
print(f"Väljund: {out_dir}\n")

try:
    with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=10) as r:
        r.read()
except Exception as e:
    print(f"Viga: server {ENDPOINT} ei vasta ({e}).")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Päring
# ---------------------------------------------------------------------------

import re


def strip_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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
        pilt = fit_to_grid(im.convert("RGB"))
        buf = io.BytesIO()
        pilt.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saada(img: Path):
    keha = json.dumps({
        "model": "vutt",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": pildi_data_uri(img)}},
        ]}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{ENDPOINT}/v1/chat/completions", data=keha,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        v = json.loads(r.read())
    return (strip_output(v["choices"][0]["message"]["content"]),
            v["choices"][0].get("finish_reason"))


TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*/?>")


def tags(t: str) -> Counter:
    return Counter(TAG_RE.findall(t))


# ---------------------------------------------------------------------------
# Jooks
# ---------------------------------------------------------------------------

rows = []
t0 = time.time()
tehtud = 0
ootel = []
for ua, teos, leht, img, txt in kandidaadid:
    p = out_dir / f"{leht}.txt"
    if RESUME and p.exists():
        rows.append((teos, leht, txt.read_text(encoding="utf-8").strip(),
                     p.read_text(encoding="utf-8").strip(), None))
    else:
        ootel.append((teos, leht, img, txt))
if rows:
    print(f"Resume: {len(rows)} lehte olemas, jäänud {len(ootel)}\n")


def too(k):
    teos, leht, img, txt = k
    try:
        uus, lopp = saada(img)
        return teos, leht, txt.read_text(encoding="utf-8").strip(), uus, lopp
    except Exception as e:
        print(f"  VIGA {leht}: {e}")
        return teos, leht, None, None, None


with ThreadPoolExecutor(max_workers=BATCH) as pool:
    for teos, leht, vana, uus, lopp in pool.map(too, ootel):
        if uus is None:
            continue
        (out_dir / f"{leht}.txt").write_text(uus, encoding="utf-8")
        rows.append((teos, leht, vana, uus, lopp))
        tehtud += 1
        if tehtud % BATCH == 0 or tehtud == len(ootel):
            k = time.time() - t0
            print(f"[{tehtud}/{len(ootel)}] {k/60:5.1f} min  ({k/tehtud:4.1f} s/lk)")

# ---------------------------------------------------------------------------
# Võrdlus
# ---------------------------------------------------------------------------

tulemused = []
tag_vana, tag_uus = Counter(), Counter()
for teos, leht, vana, uus, lopp in rows:
    tv, tu = tags(vana), tags(uus)
    tag_vana.update(tv); tag_uus.update(tu)
    tulemused.append({
        "leht": leht, "teos": teos,
        "vana_chars": len(vana), "uus_chars": len(uus),
        "ratio": round(len(uus) / max(len(vana), 1), 3),
        "erinevus": round(editdistance.eval(vana, uus) / max(len(vana), 1), 4),
        "loop_uus": "jah" if is_looped(uus) else "",
        "loop_vana": "jah" if is_looped(vana) else "",
        "lopp": lopp or "",
        "tag_vana": sum(tv.values()), "tag_uus": sum(tu.values()),
        "i_vana": tv["<i>"], "i_uus": tu["<i>"],
        "m_vana": tv["<m>"], "m_uus": tu["<m>"],
        "cs_vana": tv["<cs>"], "cs_uus": tu["<cs>"],
    })

results_csv = out_dir / "results.csv"
with open(results_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tulemused[0].keys()))
    w.writeheader(); w.writerows(tulemused)


def keskm(v):
    return sum(v) / len(v) if v else float("nan")


import statistics
er = [r["erinevus"] for r in tulemused]
ra = [r["ratio"] for r in tulemused]
print(f"\n{'lehti':>6s} {'erinevus kesk':>14s} {'mediaan':>9s} {'ratio kesk':>11s}")
print(f"{len(tulemused):6d} {keskm(er):13.1%} {statistics.median(er):8.1%} {keskm(ra):11.2f}")

print(f"\n{'märgend':10s} {'vana':>8s} {'uus':>8s} {'muut':>8s}")
for t in sorted(set(tag_vana) | set(tag_uus), key=lambda x: -tag_vana[x]):
    v, u = tag_vana[t], tag_uus[t]
    if v + u < 5:
        continue
    print(f"{t:10s} {v:8d} {u:8d} {u - v:+8d}")

loop_u = [r for r in tulemused if r["loop_uus"]]
loop_v = [r for r in tulemused if r["loop_vana"]]
katkes = [r for r in tulemused if r["lopp"] == "length"]
print(f"\nLoope: uues väljundis {len(loop_u)}, VUTT-i salvestatud tekstis {len(loop_v)}")
print(f"Tokenilakke jooksis: {len(katkes)}")
for r in loop_u:
    print(f"  LOOP uus:  {r['leht'][:44]} ratio {r['ratio']}")
for r in loop_v:
    print(f"  LOOP vana: {r['leht'][:44]} (VUTT-is praegu sees!)")

print("\nSuurima lahknevusega lehed (vaata silmaga):")
for r in sorted(tulemused, key=lambda x: -x["erinevus"])[:10]:
    print(f"  {r['erinevus']:6.1%}  ratio {r['ratio']:5.2f}  {r['leht'][:40]:40s} {r['teos'][:28]}")

k = time.time() - t0
run = {"endpoint": ENDPOINT, "materjal": MATERIAL, "since": SINCE,
       "lehti": len(tulemused), "lehti_seekord": tehtud,
       "sekundeid": round(k, 1),
       "s_per_lk": round(k / tehtud, 2) if tehtud else None,
       "lehte_tunnis": round(3600 * tehtud / k, 1) if tehtud else None,
       "aeg": time.strftime("%Y-%m-%d %H:%M")}
(out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
print(f"\nAeg: {k/60:.1f} min" + (f"  ({k/tehtud:.1f} s/lk)" if tehtud else ""))
print(f"Tulemused: {results_csv}")
