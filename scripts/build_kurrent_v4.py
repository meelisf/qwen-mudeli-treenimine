#!/usr/bin/env python3
"""Kurrendi andmestik v4 (2026-10-04) = v3 + parandatud Bullinger + DTA Kosmos + Escheri puhtad lehed.

  - alus data/kurrent_v3 (v2 miinus ≥2 tühja TextLine-iga lehed)
  - bullinger_autoren asendatakse data/bullinger_v3-ga (parim XML-versioon,
    ≤1 tühi rida); holdout-lehed jäävad, et eval oleks võrreldav
  - xix-i parthey/hufeland/nn_msgermqu (Matcheri osaline GT) asendatakse
    DTA täistekstiga: data/dta_kosmos (build_dta_kosmos.py), + 5 uut kätt
Pildid hardlink-itud. data/kurrent* sisendeid ei muudeta.
Käivitus: venv/bin/python scripts/build_kurrent_v4.py [--dry-run]
"""
import csv, os, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DRY = "--dry-run" in sys.argv
BASE = Path("data/kurrent_v3")
BULL = Path("data/bullinger_v3")
DTA = Path("data/dta_kosmos")
ESCH = Path("data/escher_lisa")
NEW = Path("data/kurrent_v4")
DTA_XIX = {"parthey", "hufeland_privatbesitz_1829", "nn_msgermqu2124_1827", "nn_msgermqu2345_1827"}

csv.field_size_limit(sys.maxsize)


def loe(d):
    rows = list(csv.reader(open(d / "metadata.csv", encoding="utf-8")))
    proj = {r["failinimi"]: r["projekt"] for r in csv.DictReader(open(d / "projektid.csv", encoding="utf-8"))}
    return rows[0], rows[1:], proj


def main():
    if NEW.exists() and not DRY:
        sys.exit(f"{NEW} on juba olemas")
    header, base, proj = loe(BASE)
    holdout = {os.path.basename(l.strip().split("\t")[-1]) for l in open(BASE / "holdout.txt", encoding="utf-8")
               if l.strip() and not l.startswith("#")}
    out, src, st = [], {}, Counter()
    for r in base:
        ho = os.path.basename(r[0]) in holdout
        if r[2] == "bullinger_autoren" and not ho:
            st["−bullinger vana"] += 1
            continue
        if proj.get(r[0]) in DTA_XIX and not ho:
            st["−xix matcher (DTA asemel)"] += 1
            continue
        out.append(r); src[r[0]] = BASE
    for d, nimi in ((BULL, "+bullinger v3"), (DTA, "+dta kosmos"), (ESCH, "+escher puhtad")):
        _, rows, p = loe(d)
        for r in rows:
            assert r[0] not in src, r[0]
            out.append(r); src[r[0]] = d; proj[r[0]] = p[r[0]]
            st[nimi] += 1
    print(f"v3 {len(base)} → v4 {len(out)}  {dict(st)}")
    print(Counter(r[2] for r in out).most_common())
    if DRY:
        return
    (NEW / "images").mkdir(parents=True)
    for r in out:
        os.link(src[r[0]] / r[0], NEW / r[0])
    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(out)
    with open(NEW / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "projekt"])
        w.writerows((r[0], proj.get(r[0], "")) for r in out)
    (NEW / "holdout.txt").write_text((BASE / "holdout.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (NEW / "SOURCE.txt").write_text(
        (BASE / "SOURCE.txt").read_text(encoding="utf-8")
        + f"\nehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_kurrent_v4.py\n"
        f"v3 {len(base)} → {len(out)}  {dict(st)}\n", encoding="utf-8")
    print(f"valmis: {NEW}")


if __name__ == "__main__":
    main()
