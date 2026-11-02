#!/usr/bin/env python3
"""Bullingeri kirjad v3 (2026-10-04): ainult täieliku GT-ga lehed.

Kaks viga vanas ehituses (build_kurrent_dataset.py --dataset …bullinger-autoren):
  1. teksti(ta) TextLine jäeti vaikselt vahele → GT-st puuduvad pildil olevad read
     (80 % treeningus olnud lehtedest ≥2 tühja rida; GT-kontrollis 35 % vs 11 %);
  2. sama fail on HF-andmestikus mitmes versioonis (eri XML), voogedastus võttis
     esimese — 100 lehel oli olemas täielikum versioon.
Siin: iga faili kohta parim versioon (vähim tühje ridu), alles ainult ≤ MAX_TUHJE
tühja reaga lehed. Tekst sama `parse_pagexml`-iga nagu varem.

Pildid: juba data/kurrent/images-is olevad hardlink'itakse (pilt ei sõltu XML-i
versioonist); ülejäänud loetakse HF-ist (üks row group faili kohta → terve fail).
Holdout'i lehed jäetakse välja.

Väljund: data/bullinger_v3/{images/, metadata.csv, projektid.csv, SOURCE.txt}
Käivitus: venv/bin/python scripts/build_bullinger_v3.py [--dry-run]
"""
import csv, io, os, re, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_kurrent_dataset import parse_pagexml                     # noqa: E402

DRY = "--dry-run" in sys.argv
REPO = "datasets/dh-unibe/image-text_bullinger-autoren"
OLD = Path("data/kurrent")
OUT = Path("data/bullinger_v3")
ALLIKAS = "bullinger_autoren"
MIN_LINES = 5                # = build_kurrent_dataset.MIN_LINES
MAX_TUHJE = 1
LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)

csv.field_size_limit(sys.maxsize)


def tuhje(xml):
    n = e = 0
    for tl in LINE.findall(xml):
        u = UNI.findall(WORD.sub("", tl))
        n += 1
        e += not (u and u[-1].strip())
    return n, e


def stem_of(filename):
    return re.sub(r"[^\w\-]", "_", Path(filename).stem)


def main():
    if OUT.exists() and not DRY:
        sys.exit(f"{OUT} on juba olemas")
    have = {}
    for r in csv.reader(open(OLD / "metadata.csv", encoding="utf-8")):
        m = re.match(r"images/\d+_bullinger_au_(.+)\.jpg$", r[0])
        if m:
            have[m.group(1)] = r[0]
    holdout = {m.group(1) for l in open(OLD / "holdout.txt", encoding="utf-8")
               if (m := re.search(r"_bullinger_au_(.+)\.jpg", l))}

    fs = HfFileSystem()
    files = sorted(fs.glob(f"{REPO}/data/**/*.parquet"))
    best = {}                    # tüvi → (tühje, tekst, parquet-fail, projekt)
    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"xml {i}/{len(files)}", flush=True)
        for b in pq.ParquetFile(fs.open(f, block_size=1 << 20)).iter_batches(
                batch_size=64, columns=["xml_content", "filename", "project_name"]):
            for r in b.to_pylist():
                x = r["xml_content"] or ""
                _, e = tuhje(x)
                text = parse_pagexml(x)
                if not text or sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES:
                    continue
                s = stem_of(r["filename"])
                if s not in best or e < best[s][0]:
                    best[s] = (e, text, f, r["project_name"])
    stats = Counter()
    valik = {}
    for s, (e, text, f, p) in best.items():
        if s in holdout:
            stats["holdout"] += 1
        elif e > MAX_TUHJE:
            stats["tühjad read"] += 1
        else:
            valik[s] = (text, f, p)
            stats["olemas pilt" if s in have else "uus pilt"] += 1
    print(f"unikaalseid: {len(best)}  {dict(stats)}  → {len(valik)}", flush=True)
    if DRY:
        return

    (OUT / "images").mkdir(parents=True)
    rows = []
    vaja = {}                    # parquet-fail → {tüvi}
    for s, (text, f, p) in valik.items():
        name = f"images/blv3_{s}.jpg"
        if s in have:
            os.link(OLD / have[s], OUT / name)
            rows.append((name, text, ALLIKAS, p))
        else:
            vaja.setdefault(f, set()).add(s)
    print(f"pilte HF-ist: {sum(len(v) for v in vaja.values())} lk {len(vaja)} failist", flush=True)
    for f, stems in vaja.items():
        for b in pq.ParquetFile(fs.open(f, block_size=8 << 20)).iter_batches(
                batch_size=4, columns=["image", "filename"]):
            for r in b.to_pylist():
                s = stem_of(r["filename"])
                if s not in stems:
                    continue
                stems.discard(s)
                name = f"images/blv3_{s}.jpg"
                try:
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(
                        OUT / name, "JPEG", quality=90)
                except Exception as e:
                    print(f"  pildi viga {s}: {e}", flush=True)
                    continue
                text, _, p = valik[s]
                rows.append((name, text, ALLIKAS, p))
        print(f"  {f.split('/')[-2]} tehtud, puudu {len(stems)}", flush=True)
    rows.sort()
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["failinimi", "transkriptsioon", "allikas"])
        w.writerows(r[:3] for r in rows)
    with open(OUT / "projektid.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["failinimi", "projekt"])
        w.writerows((r[0], r[3]) for r in rows)
    (OUT / "SOURCE.txt").write_text(
        f"ehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_bullinger_v3.py\n"
        f"dh-unibe/image-text_bullinger-autoren, parim XML-versioon faili kohta, ≤{MAX_TUHJE} tühja TextLine'i\n"
        f"lehti: {len(rows)}  {dict(stats)}\n", encoding="utf-8")
    print(f"valmis: {OUT} ({len(rows)} lk)")


if __name__ == "__main__":
    main()
