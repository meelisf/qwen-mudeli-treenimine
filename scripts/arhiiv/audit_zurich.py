#!/usr/bin/env python3
"""Zürichi (kurrent_xix MM_1_*) v4 lehtede tühjad TextLine'id HF-ist, ainult xml_content.

Kõik 1 000 v4 Zürichi lehte on CITlabi Matcheri automaatjoondus; 999-l on ≥2
tühja TextLine-i (04.10). Väljund data/kurrent_xix_audit/tuhjad_read_zurich.csv.
Käivitus: venv/bin/python scripts/audit_zurich.py   (~2 h, HF range-päringud)
"""
import csv, re, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

csv.field_size_limit(sys.maxsize)
LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)
CREATOR = re.compile(r"<Creator>(.*?)</Creator>", re.S)

want = {}
for r in csv.DictReader(open("data/kurrent_v4/metadata.csv", encoding="utf-8")):
    if r["allikas"] == "kurrent_xix_zurich":
        m = re.match(r"images/\d+_(.+)\.jpg$", r["failinimi"])
        want[m.group(1)] = r["failinimi"]
print("v4 Zürich", len(want), flush=True)
prefixes = {k.split("_")[0] for k in want}
fs = HfFileSystem()
files = sorted(fs.glob("datasets/dh-unibe/image-text_kurrent-xix/data/train/MM_1_0*/*.parquet"))
print("faile", len(files), flush=True)
out = open("data/kurrent_xix_audit/tuhjad_read_zurich.csv", "w", newline="", encoding="utf-8")
w = csv.writer(out)
w.writerow(["projekt", "failinimi", "ridu", "tuhje", "looja", "treeningus"])
leitud = 0
for i, f in enumerate(files):
    for b in pq.ParquetFile(fs.open(f, block_size=1 << 20)).iter_batches(
            batch_size=64, columns=["xml_content", "filename"]):
        for r in b.to_pylist():
            s = re.sub(r"[^\w\-]", "_", Path(r["filename"]).stem)
            if s not in want:
                continue
            x = r["xml_content"] or ""
            n = e = 0
            for tl in LINE.findall(x):
                u = UNI.findall(WORD.sub("", tl))
                n += 1
                e += not (u and u[-1].strip())
            c = CREATOR.search(x)
            lj = "matcher" if c and "Matcher" in c.group(1) else (c.group(1)[:30].replace("\n", " ") if c else "?")
            w.writerow([f.split("/")[-2], r["filename"], n, e, lj, want[s]])
            leitud += 1
    print(f"{i+1}/{len(files)} leitud {leitud}", flush=True)
out.close()
