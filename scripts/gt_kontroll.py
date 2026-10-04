#!/usr/bin/env python3
"""
Kurrendi treeningmaterjali kiirkontroll (SEIS §6.11 tee c)

Laseb tootmismudeli üle KOGU treeningkomplekti (metadata.csv miinus holdout)
ja võrdleb väljundit GT-ga. Kõrge CER = kas raske leht VÕI vale GT; skript
teeb ainult KANDIDAATIDE nimekirja, otsus jääb inimesele.

  python scripts/gt_kontroll.py --dry-run          # nimekiri, serverit ei puudu
  python scripts/gt_kontroll.py --limit 20         # proov
  python scripts/gt_kontroll.py                    # täisjooks (jätkab ise pooleli jäänut)
  python scripts/gt_kontroll.py --aruanne          # ainult aruanne kettal olevast
  python scripts/gt_kontroll.py --naita <tüvi>     # GT ja väljund ühe lehe kohta

Vaikimisi tootmisserver :8081 (llama-server-hand, -np 4) ja 3 paralleelset
päringut — üks slot jääb teenusele, nii et VUTT-i OCR töötab jooksu ajal
edasi (aeglasemalt). Öösel võib anda `--paralleel 4`.

PIIR: mudel on neid lehti treeningul 2 epohhi näinud ja on osa GT vigu sisse
õppinud. Tuvastub JÄME viga (vale lehe tekst, kärbitud/poolik GT, rämps-HTR,
mida mudel ei suutnud omandada). Madal CER EI tõenda, et GT on õige.

Väljund: data/kurrent/gt_kontroll/<nimi>/
  valjundid/<tüvi>.txt   mudeli väljund (olemasolu = tehtud, jätkamise alus)
  meta.jsonl             finish_reason, sekundid lehe kohta
  tulemused.csv          kõik lehed mõõdikutega
  kandidaadid.csv        märgitud lehed, halvim ees
  kokkuvote.md           allika kaupa
"""

import argparse
import base64
import csv
import io
import json
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from imaging import fit_to_grid
from loop_detect import is_looped
from prompt import KURRENT_INSTRUCTION
from textmetrics import cer, normaliseeri, strip_output

csv.field_size_limit(10 ** 7)

DATA_CSV     = Path("data/kurrent/metadata.csv")
DATA_IMAGES  = Path("data/kurrent/images")
HOLDOUT_PATH = Path("data/kurrent/holdout.txt")
OUT_ROOT     = Path("data/kurrent/gt_kontroll")

MAX_NEW_TOKENS = 4096
# Nii palju järjestikuseid päringuvigu = server on maas; öösel ei tohi skript
# tühja ringi käia ja tuhandeid lehti „vigaseks" märkida.
MAX_JAREST_VIGU = 20

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--endpoint", default="http://127.0.0.1:8081")
ap.add_argument("--nimi", default="kurrent-20261002-Q8_0")
ap.add_argument("--paralleel", type=int, default=3)
ap.add_argument("--limit", type=int)
ap.add_argument("--lavi", type=float, default=0.10,
                help="normaliseeritud CER, millest alates leht on kandidaat")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--aruanne", action="store_true")
ap.add_argument("--naita")
args = ap.parse_args()

out_dir = OUT_ROOT / args.nimi
val_dir = out_dir / "valjundid"

# ---------------------------------------------------------------------------
# Nimekiri: täpselt see, mida train_kurrent.py nägi (v.a üle MAX_SEQ lehed —
# neid on 2 ja tokenisaatorit ainult nende pärast ei laeta)
# ---------------------------------------------------------------------------

holdout = set()
for line in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        holdout.add(Path(line.split("\t")[-1]).name)

lehed = []   # (tüvi, failinimi, allikas, gt)
puuduvad = 0
with open(DATA_CSV, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        nimi = Path(row["failinimi"]).name
        gt = (row.get("transkriptsioon") or "").strip()
        if not gt or nimi in holdout:
            continue
        if not (DATA_IMAGES / nimi).exists():
            puuduvad += 1
            continue
        lehed.append((Path(nimi).stem, nimi, row.get("allikas") or "?", gt))

tyved = [t for t, *_ in lehed]
if len(set(tyved)) != len(tyved):
    sys.exit("Viga: failitüved ei ole unikaalsed — väljundid kirjutaksid üksteist üle")

# Segatud järjekord: pooleli jäänud jooks katab siis kõiki allikaid ühtlaselt,
# mitte esimest 40 % CSV-st.
random.Random(3407).shuffle(lehed)
if args.limit:
    lehed = lehed[:args.limit]

# ---------------------------------------------------------------------------
# --naita
# ---------------------------------------------------------------------------

if args.naita:
    for tyvi, nimi, allikas, gt in lehed:
        if tyvi == args.naita or nimi == args.naita:
            p = val_dir / f"{tyvi}.txt"
            print(f"# {nimi}  ({allikas})  pilt: {DATA_IMAGES / nimi}\n")
            print("=== GT ===\n" + gt + "\n")
            print("=== MUDEL ===\n" + (p.read_text(encoding="utf-8") if p.exists() else "(puudub)"))
            sys.exit(0)
    sys.exit(f"Lehte {args.naita} ei ole nimekirjas")

# ---------------------------------------------------------------------------
# Aruanne (kutsutakse ka jooksu lõpus)
# ---------------------------------------------------------------------------


def lae_meta():
    meta = {}
    p = out_dir / "meta.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                meta[m["tyvi"]] = m
    return meta


def liigita(r):
    """Miks leht kandidaat on. Järjekord = usaldusväärsus."""
    if r["loop"] or r["finish"] == "length":
        return "loop"            # mudeli viga; GT võib olla korras
    if r["ratio"] < 0.7:
        return "GT pikem"        # GT-s teksti, mida pildil pole? (vale leht, naaberleht)
    if r["ratio"] > 1.4:
        return "GT lühem"        # poolik / kärbitud GT
    if r["cer_norm"] >= args.lavi:
        return "kõrge CER"
    return ""


def aruanne():
    meta = lae_meta()
    rows = []
    for tyvi, nimi, allikas, gt in lehed:
        p = val_dir / f"{tyvi}.txt"
        if not p.exists():
            continue
        hyp = p.read_text(encoding="utf-8")
        rn, hn = normaliseeri(gt), normaliseeri(hyp)
        r = {
            "failinimi": nimi, "allikas": allikas,
            "gt_chars": len(gt), "out_chars": len(hyp),
            "ratio": round(len(hyp) / max(len(gt), 1), 3),
            "cer": round(cer(gt, hyp), 4),
            "cer_norm": round(cer(rn, hn), 4),
            "loop": bool(is_looped(hyp)),
            "finish": meta.get(tyvi, {}).get("finish", ""),
        }
        r["liik"] = liigita(r)
        r["gt_algus"] = gt[:80].replace("\n", " ⏎ ")
        r["out_algus"] = hyp[:80].replace("\n", " ⏎ ")
        rows.append(r)
    if not rows:
        print("Aruanne: ühtegi väljundit veel ei ole.")
        return

    valjad = list(rows[0].keys())
    with open(out_dir / "tulemused.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=valjad)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["allikas"], r["failinimi"])))
    kand = sorted((r for r in rows if r["liik"]), key=lambda r: -r["cer_norm"])
    with open(out_dir / "kandidaadid.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=valjad)
        w.writeheader()
        w.writerows(kand)

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]

    liigid = ["loop", "GT pikem", "GT lühem", "kõrge CER"]
    md = [f"# GT kiirkontroll — {args.nimi}", "",
          f"Lehti tehtud: {len(rows)} / {len(lehed)}.  Kandidaate: {len(kand)} "
          f"({len(kand) / len(rows):.1%}).  Lävi: normaliseeritud CER ≥ {args.lavi:.0%}.", "",
          "Kõrge mediaan TERVEL allikal = kahtlane allikas (nt rämps-HTR), mitte üksikleht.", "",
          "| allikas | lk | med CER norm | p90 | " + " | ".join(liigid) + " | kandidaate |",
          "|---|---|---|---|" + "---|" * len(liigid) + "---|"]
    for allikas in sorted({r["allikas"] for r in rows}):
        g = [r for r in rows if r["allikas"] == allikas]
        c = [r["cer_norm"] for r in g]
        k = sum(1 for r in g if r["liik"])
        md.append(f"| {allikas} | {len(g)} | {statistics.median(c):.1%} | {pct(c, 0.9):.1%} | "
                  + " | ".join(str(sum(1 for r in g if r["liik"] == l)) for l in liigid)
                  + f" | {k} ({k / len(g):.0%}) |")
    c = [r["cer_norm"] for r in rows]
    md.append(f"| **KOKKU** | {len(rows)} | {statistics.median(c):.1%} | {pct(c, 0.9):.1%} | "
              + " | ".join(str(sum(1 for r in rows if r["liik"] == l)) for l in liigid)
              + f" | {len(kand)} |")
    md += ["", "## Halvimad 30", "",
           "| failinimi | allikas | liik | CER norm | ratio |", "|---|---|---|---|---|"]
    for r in kand[:30]:
        md.append(f"| {r['failinimi']} | {r['allikas']} | {r['liik']} | "
                  f"{r['cer_norm']:.1%} | {r['ratio']:.2f} |")
    md += ["", "Lehe vaatamine: `python scripts/gt_kontroll.py --naita <tüvi>`"]
    (out_dir / "kokkuvote.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:md.index("## Halvimad 30")]))
    print(f"\nFailid: {out_dir}/{{kokkuvote.md,kandidaadid.csv,tulemused.csv}}")


print(f"Server:  {args.endpoint}  (paralleelseid: {args.paralleel})")
print(f"Lehti:   {len(lehed)}  (holdout {len(holdout)} väljas, pilt puudu {puuduvad})")
print(f"Väljund: {out_dir}")

if args.dry_run:
    from collections import Counter
    for allikas, n in sorted(Counter(a for _, _, a, _ in lehed).items()):
        print(f"  {allikas:28s} {n:6d}")
    sys.exit(0)

if args.aruanne:
    aruanne()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Jooks
# ---------------------------------------------------------------------------

try:
    with urllib.request.urlopen(f"{args.endpoint}/health", timeout=10) as r:
        r.read()
except Exception as e:
    sys.exit(f"Viga: server {args.endpoint} ei vasta ({e})")

val_dir.mkdir(parents=True, exist_ok=True)
ootel = [x for x in lehed if not (val_dir / f"{x[0]}.txt").exists()]
print(f"Tehtud varem: {len(lehed) - len(ootel)}, ootel: {len(ootel)}\n")


def pildi_data_uri(path: Path) -> str:
    # PNG ja fit_to_grid nagu eval_kurrent.py-s (põhjendus seal): JPEG-i
    # ümberkodeerimine sööb peeneid jooni, võrk peab olema treeningu oma.
    with PILImage.open(path) as im:
        pilt = fit_to_grid(im.convert("RGB"), resample=PILImage.LANCZOS)
        buf = io.BytesIO()
        pilt.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def too(kirje):
    tyvi, nimi, _, _ = kirje
    t = time.time()
    keha = json.dumps({
        "model": "kurrent",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": KURRENT_INSTRUCTION},
            {"type": "image_url", "image_url": {"url": pildi_data_uri(DATA_IMAGES / nimi)}},
        ]}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{args.endpoint}/v1/chat/completions", data=keha,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        vastus = json.loads(r.read())
    valik = vastus["choices"][0]
    return tyvi, strip_output(valik["message"]["content"]), valik.get("finish_reason"), time.time() - t


t0 = time.time()
tehtud = vigu = jarest_vigu = 0
peatus = False

with ThreadPoolExecutor(max_workers=args.paralleel) as pool, \
        open(out_dir / "meta.jsonl", "a", encoding="utf-8") as metaf:
    tulevikud = {pool.submit(too, x): x for x in ootel}
    for fut in as_completed(tulevikud):
        if peatus:
            fut.cancel()
            continue
        try:
            tyvi, hyp, finish, sek = fut.result()
        except Exception as e:
            vigu += 1
            jarest_vigu += 1
            print(f"  VIGA {tulevikud[fut][1]}: {e}")
            if jarest_vigu >= MAX_JAREST_VIGU:
                print(f"\nSTOPP: {MAX_JAREST_VIGU} viga järjest — server maas? "
                      "Käivita uuesti, jätkab pooleli jäänust.")
                peatus = True
                for f in tulevikud:
                    f.cancel()
            continue
        jarest_vigu = 0
        # Kõigepealt meta, siis .txt: .txt olemasolu tähendab „tehtud"
        metaf.write(json.dumps({"tyvi": tyvi, "finish": finish, "s": round(sek, 1)}) + "\n")
        metaf.flush()
        (val_dir / f"{tyvi}.txt").write_text(hyp, encoding="utf-8")
        tehtud += 1
        if tehtud % 50 == 0 or tehtud == len(ootel):
            kulunud = time.time() - t0
            jaanud = (len(ootel) - tehtud) * kulunud / tehtud
            print(f"[{tehtud}/{len(ootel)}] {kulunud / 3600:5.2f} h, "
                  f"{kulunud / tehtud:4.2f} s/lk, jäänud ~{jaanud / 3600:.1f} h, vigu {vigu}",
                  flush=True)

print(f"\nTehtud {tehtud}, vigu {vigu}, aeg {(time.time() - t0) / 3600:.2f} h")
aruanne()
if peatus or vigu:
    sys.exit(1)
