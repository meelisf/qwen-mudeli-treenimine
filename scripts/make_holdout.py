#!/usr/bin/env python3
"""
Kurrent-andmestiku holdout-nimekirja koostamine

Valib igast allikast väikese hulga lehti, mis jäävad treeningust VÄLJA ja
mille peal saab mudeleid omavahel võrrelda. Nimekiri on determinstlik
(seed) ja läheb repositooriumi, et sama komplekt kehtiks ka aasta pärast.

Kogus on meelega väike (~70 lk, 0,4 % andmestikust): eesmärk on mõõdik,
mitte statistiliselt range hindamine. Iga allikas peab olema esindatud,
sest CER erineb allikate lõikes rohkem kui mudelite lõikes.

Käivitamine:
  python scripts/make_holdout.py --stats     # ainult näita, ei kirjuta
  python scripts/make_holdout.py             # kirjutab data/kurrent/holdout.txt

Kasutajad:
  train_kurrent.py   – jätab need read treeningust välja
  eval_kurrent.py    – laseb mudeli nende lehtede peale ja arvutab CER-i
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 7)

DATA_CSV = Path("data/kurrent/metadata.csv")
OUT_PATH = Path("data/kurrent/holdout.txt")

SEED = 3407          # sama seeme mis treeningul – juhus on siin ainult jaotur
PCT = 0.006          # osakaal allikast
MIN_PER_SOURCE = 3   # ka kõige väiksem allikas peab esindatud olema
MAX_PER_SOURCE = 10  # kurrent_xix (8000 lk) ei tohi komplekti ära täita
MIN_CHARS = 200      # liiga lühike leht ei anna CER-ile alust

# Tühjad ja hõredad VUTT-i lehed jäävad treeningusse – neid on 26 ja iga
# eksemplar loeb. Tühja lehe käitumist saab kontrollida ilma GT-ta:
# võta VUTT-ist märgendamata tühi versopool ja vaata, mida mudel ütleb.
SKIP_SOURCES = {"vutt_horedad"}

DRY_RUN = "--stats" in sys.argv

for i, a in enumerate(sys.argv):
    if a == "--seed" and i + 1 < len(sys.argv):
        SEED = int(sys.argv[i + 1])
    elif a == "--out" and i + 1 < len(sys.argv):
        OUT_PATH = Path(sys.argv[i + 1])

if not DATA_CSV.exists():
    print(f"Viga: andmestik puudub: {DATA_CSV}")
    sys.exit(1)

by_source = defaultdict(list)
with open(DATA_CSV, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 2:
            continue
        allikas = row[2] if len(row) >= 3 and row[2] else "(sildita)"
        if allikas in SKIP_SOURCES:
            continue
        if len(row[1].strip()) < MIN_CHARS:
            continue
        by_source[allikas].append(row[0])

rng = random.Random(SEED)
selected = []
print(f"{'allikas':28s} {'kõlblikke':>10s} {'valitud':>8s}")
for allikas in sorted(by_source):
    files = sorted(by_source[allikas])          # sorteeri enne – kettajärjekord ei tohi lugeda
    n = min(MAX_PER_SOURCE, max(MIN_PER_SOURCE, round(len(files) * PCT)))
    n = min(n, len(files))
    pick = rng.sample(files, n)
    selected.extend((allikas, p) for p in sorted(pick))
    print(f"{allikas:28s} {len(files):10d} {n:8d}")

total_rows = sum(len(v) for v in by_source.values())
print(f"{'KOKKU':28s} {total_rows:10d} {len(selected):8d}"
      f"  = {len(selected) / total_rows * 100:.2f}% kõlblikest")

if DRY_RUN:
    print("\n--stats: faili ei kirjutata.")
    sys.exit(0)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"# Kurrent holdout – treeningust välja jäetud lehed\n")
    f.write(f"# Loodud: scripts/make_holdout.py (seed={SEED}, pct={PCT}, "
            f"min={MIN_PER_SOURCE}, max={MAX_PER_SOURCE}, min_chars={MIN_CHARS})\n")
    f.write(f"# Lehti: {len(selected)}\n")
    f.write("# Vorming: <allikas>\\t<failinimi>\n")
    for allikas, failinimi in selected:
        f.write(f"{allikas}\t{failinimi}\n")

print(f"\nKirjutatud: {OUT_PATH} ({len(selected)} lehte)")
print("Järgmine samm: treening jätab need ise vahele; võrdluseks")
print("  python scripts/eval_kurrent.py models/qwen3.5-ocr-kurrent-20260602")
