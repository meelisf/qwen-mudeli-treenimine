#!/usr/bin/env python3
"""Kurrendi andmestik v3 (2026-10-04): v2 miinus osalise GT-ga lehed.

`parse_pagexml` jätab teksti(ta) TextLine'i vaikselt vahele. CITlabi „Matcher"
(editsioonitekst automaatjoondatud ridadele) jätab ebaõnnestunud joonduse
tühjaks → GT-st puuduvad read, mis pildil on. Mõõdik:
`scripts/tuhjad_read_audit.py` → `data/kurrent_xix_audit/tuhjad_read.csv`.
GT-kontrollis: ≥2 tühja rida → 58 % kandidaadid, 0–1 → 3–5 % (= taust).

Ehitab UUE kausta `data/kurrent_v3/` (pildid hardlink'itud). `data/kurrent/`
jääb PUUTUMATA. Holdout'ist eemaldatakse samad lehed (ei ole ka treeningus).
Vahetus on eraldi käsitsi samm.

Käivitus:
  venv/bin/python scripts/build_kurrent_v3.py --dry-run
  venv/bin/python scripts/build_kurrent_v3.py
"""
import csv, os, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DRY = "--dry-run" in sys.argv
LAVI = 2                                   # tühjade TextLine'ide arv, millest leht välja
OLD = Path("data/kurrent")
NEW = Path("data/kurrent_v3")
AUDIT = Path("data/kurrent_xix_audit/tuhjad_read.csv")

csv.field_size_limit(sys.maxsize)


def main():
    if NEW.exists() and not DRY:
        sys.exit(f"{NEW} on juba olemas")
    halvad = {r["treeningus"]: r for r in csv.DictReader(open(AUDIT, encoding="utf-8"))
              if r["treeningus"] and int(r["tuhje"]) >= LAVI}

    rows = list(csv.reader(open(OLD / "metadata.csv", encoding="utf-8")))
    header, rows = rows[0], rows[1:]
    keep = [r for r in rows if r[0] not in halvad]
    eemaldatud = Counter(halvad[r[0]]["projekt"] for r in rows if r[0] in halvad)
    allikad = Counter(r[2] for r in rows if r[0] in halvad)

    ho_read, ho_valja = [], []
    for line in open(OLD / "holdout.txt", encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#") and ("images/" + os.path.basename(s.split("\t")[-1])) in halvad:
            ho_valja.append(s)
        else:
            ho_read.append(line)

    print(f"read: {len(rows)} → {len(keep)} (−{len(rows) - len(keep)})")
    print(f"holdout: −{len(ho_valja)}: {', '.join(ho_valja)}")
    print("allika kaupa:", dict(allikad.most_common()))
    print("projekti kaupa:", dict(eemaldatud.most_common()))
    if DRY:
        return

    (NEW / "images").mkdir(parents=True)
    for r in keep:
        os.link(OLD / r[0], NEW / r[0])
    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(keep)
    keep_names = {r[0] for r in keep}
    with open(OLD / "projektid.csv", encoding="utf-8") as fi, \
         open(NEW / "projektid.csv", "w", newline="", encoding="utf-8") as fo:
        rd, w = csv.reader(fi), csv.writer(fo)
        w.writerow(next(rd))
        w.writerows(r for r in rd if r[0] in keep_names)
    with open(NEW / "holdout.txt", "w", encoding="utf-8") as f:
        f.write(f"# v3: eemaldatud {len(ho_valja)} osalise GT-ga lehte (≥{LAVI} tühja TextLine'i)\n")
        f.writelines(ho_read)
    src = (OLD / "SOURCE.txt").read_text(encoding="utf-8")
    (NEW / "SOURCE.txt").write_text(
        src + f"\nehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_kurrent_v3.py\n"
        f"read: {len(rows)} → {len(keep)}  välja lehed ≥{LAVI} tühja TextLine'iga "
        f"(tuhjad_read.csv); holdout −{len(ho_valja)}\n", encoding="utf-8")
    print(f"valmis: {NEW}")


if __name__ == "__main__":
    main()
