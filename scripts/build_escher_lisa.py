#!/usr/bin/env python3
"""Escheri (TRAIN_CITlab_Escher_M1) puhtad lehed, mis v3-s veel ei ole (2026-10-04).

v2 piiras Escheri 1 000 juhusliku leheni; v3 viskas neist osalise GT-ga
(≥2 tühja TextLine-i, CITlabi Matcher) välja → alles 201. Kasutaja vaatas
pildid üle: erinevad käed, VUTT-ile lähedased → võtame KÕIK puhtad
unikaalsed lehed (≤1 tühi rida, ≥5 täidetud rida; GT-kontrollis 0/17 kandidaati).
Tekst sama parse_pagexml-iga nagu v2. Väljund data/escher_lisa/.
Käivitus: venv/bin/python scripts/build_escher_lisa.py [--dry-run]
"""
import csv, glob, io, re, sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_kurrent_dataset import parse_pagexml                     # noqa: E402

DRY = "--dry-run" in sys.argv
PROJ = "TRAIN_CITlab_Escher_M1"
ALLIKAS = "xix_read_1850_99"
XIX = glob.glob(str(Path.home() / "_kustutamiseks_20261002/hf_cache/datasets--dh-unibe--image-text_kurrent-xix/snapshots/*/data/train"))[0]
OUT = Path("data/escher_lisa")
MIN_LINES = 5
csv.field_size_limit(sys.maxsize)


def main():
    if OUT.exists() and not DRY:
        sys.exit(f"{OUT} on juba olemas")
    uniq = {r["filename"] for r in csv.DictReader(open("data/kurrent_xix_audit/unique_pages.csv", encoding="utf-8"))
            if r["project"] == PROJ}
    olemas = {re.sub(r"^images/\d+_xixr_", "", r["failinimi"])[:-4]
              for r in csv.DictReader(open("data/kurrent_v3/projektid.csv", encoding="utf-8")) if r["projekt"] == PROJ}
    holdout = {m.group(1) for l in open("data/kurrent/holdout.txt", encoding="utf-8")
               if (m := re.search(r"_xixr_(\S+)\.jpg", l))}
    puhas = {x["failinimi"] for x in csv.DictReader(open("data/kurrent_xix_audit/tuhjad_read.csv", encoding="utf-8"))
             if x["projekt"] == PROJ and x["failinimi"] in uniq and int(x["tuhje"]) <= 1}
    stem = lambda f: re.sub(r"[^\w\-]", "_", Path(f).stem)[:60]
    want = {f for f in puhas if stem(f) not in olemas and stem(f) not in holdout}
    print(f"unikaalseid {len(uniq)}, puhtaid {len(puhas)}, v3-s {len(olemas)}, lisada kuni {len(want)}", flush=True)
    rows = []
    if not DRY:
        (OUT / "images").mkdir(parents=True)
    for pf in sorted(glob.glob(f"{XIX}/{PROJ}/*.parquet")):
        cols = ["xml_content", "filename"] + ([] if DRY else ["image"])
        for b in pq.ParquetFile(pf).iter_batches(batch_size=16, columns=cols):
            for r in b.to_pylist():
                if r["filename"] not in want:
                    continue
                want.discard(r["filename"])
                text = parse_pagexml(r["xml_content"] or "")
                if not text or sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES:
                    continue
                name = f"images/esch_{stem(r["filename"])}.jpg"
                if not DRY:
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(OUT / name, "JPEG", quality=90)
                rows.append((name, text, ALLIKAS))
    print(f"lisatud {len(rows)}, leidmata {len(want)}")
    if DRY:
        return
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "transkriptsioon", "allikas"]); w.writerows(rows)
    with open(OUT / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "projekt"]); w.writerows((r[0], PROJ) for r in rows)
    (OUT / "SOURCE.txt").write_text(f"scripts/build_escher_lisa.py: {len(rows)} puhast Escheri lehte (≤1 tühi TextLine)\n", encoding="utf-8")
    print(f"valmis: {OUT}")


if __name__ == "__main__":
    main()
