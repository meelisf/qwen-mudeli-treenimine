"""Kurrendi v4: GT-kontrolli järgi vigase GT-ga lehed metadata.csv-st välja (2026-10-06).

Sisend: data/kurrent/gt_kontroll/<nimi>/valja.csv (scripts/gt_sonakate.py).
Eemaldab read failidest metadata.csv ja projektid.csv; pildid jäävad kausta
(treening loeb ainult metadata.csv-d). Holdout jääb puutumata — kui mõni
väljaviidav leht on holdoutis, peatub. vutt_horedad ei tohi nimekirjas olla
(tahtlikult peaaegu tühjad lehed, vt gt_sonakate.py).

Vaikimisi kuivkäivitus. Kirjutamine: --kirjuta (varukoopia metadata.csv.bak).
"""
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

csv.field_size_limit(10 ** 7)

K = Path("data/kurrent")
VALJA = K / "gt_kontroll/kurrent-20261002-Q8_0/valja.csv"
KIRJUTA = "--kirjuta" in sys.argv

valja = {r["failinimi"]: r for r in csv.DictReader(open(VALJA, encoding="utf-8"))}
holdout = {os.path.basename(l.strip().split("\t")[-1])
           for l in open(K / "holdout.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}

with open(K / "metadata.csv", newline="", encoding="utf-8") as f:
    rd = csv.reader(f)
    header = next(rd)
    read = list(rd)
nimi = lambda r: os.path.basename(r[0])
olemas = {nimi(r) for r in read}

vead = []
if valja.keys() & holdout:
    vead.append(f"holdoutis: {sorted(valja.keys() & holdout)}")
if any(r["allikas"] == "vutt_horedad" for r in valja.values()):
    vead.append("vutt_horedad on nimekirjas")
if valja.keys() - olemas:
    vead.append(f"metadata.csv-s puudu {len(valja.keys() - olemas)} (juba eemaldatud?): "
                f"{sorted(valja.keys() - olemas)[:5]}")
if vead:
    sys.exit("PEATUS:\n  " + "\n  ".join(vead))

jaab = [r for r in read if nimi(r) not in valja]
treeningule = sum(1 for r in jaab if nimi(r) not in holdout)
print(f"metadata.csv {len(read)} → {len(jaab)} (−{len(read) - len(jaab)}); "
      f"treeningule {treeningule}, holdout {len(holdout)}")
print("põhjused:", dict(Counter(r["pohjus"] for r in valja.values()).most_common()))
print("allikad: ", dict(Counter(r["allikas"] for r in valja.values()).most_common()))
if not KIRJUTA:
    print("\nKuivkäivitus. Kirjutamiseks: --kirjuta")
    sys.exit(0)

shutil.copy2(K / "metadata.csv", K / "metadata.csv.bak")
with open(K / "metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(header); w.writerows(jaab)

with open(K / "projektid.csv", newline="", encoding="utf-8") as f:
    rd = csv.reader(f)
    p_header = next(rd)
    p_read = [r for r in rd if os.path.basename(r[0]) not in valja]
with open(K / "projektid.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(p_header); w.writerows(p_read)

with open(K / "SOURCE.txt", "a", encoding="utf-8") as f:
    f.write(f"\nGT-kontroll: {datetime.now().isoformat(timespec='seconds')}  scripts/kurrent_gt_valja.py\n"
            f"read: {len(read)} → {len(jaab)}  välja {len(valja)} ({VALJA}; "
            f"{dict(Counter(r['pohjus'] for r in valja.values()))})\n")
print("Kirjutatud. Varukoopia: data/kurrent/metadata.csv.bak")
