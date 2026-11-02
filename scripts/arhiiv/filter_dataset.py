#!/usr/bin/env python3
"""
Filtreerib metadata.csv allikate järgi.

Kasutamine:
  # Näita allikate loetelu ja mahtu
  python scripts/filter_dataset.py --stats

  # Jäta välja konkreetsed allikad
  python scripts/filter_dataset.py --exclude kurrent_xix --out data/kurrent/metadata_filtered.csv

  # Kaasa ainult konkreetsed allikad
  python scripts/filter_dataset.py --include svea_hovratt_seg trolldomskommissionen_seg hanse_kurrent_xvi

  # Tasakaalusta: maksimaalselt N näidet allikast
  python scripts/filter_dataset.py --max-per-source 500 --out data/kurrent/metadata_balanced.csv

  # Kombinatsioon
  python scripts/filter_dataset.py --exclude kurrent_xix --max-per-source 1000 --out data/kurrent/metadata_filtered.csv

Vaikimisi kirjutatakse --out faili; kui --out puudub, prinditakse statistika.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path("data/kurrent/metadata.csv")


def _args(flag):
    result = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == flag:
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                result.append(sys.argv[i])
                i += 1
        else:
            i += 1
    return result

def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


EXCLUDE        = set(_args("--exclude"))
INCLUDE        = set(_args("--include"))
MAX_PER_SOURCE = int(_arg("--max-per-source", 0))
OUT_PATH       = _arg("--out")
STATS_ONLY     = "--stats" in sys.argv
SEED           = int(_arg("--seed", 42))


def detect_allikas(failinimi: str) -> str:
    """Sama loogika mis add_allikas_column.py-s."""
    import re
    name = Path(failinimi).name
    rest = re.sub(r"^\d{5}_", "", name)
    if re.match(r"^08\d{4}_", rest):      return "kurrent_xix"
    if rest.startswith("aaeb_"):           return "aaeb_xiv_xvii"
    if rest.startswith("bullinger_"):      return "bullinger_autoren"
    if rest.startswith("koenigsfelde"):    return "koenigsfelden_adhr"
    if rest.startswith("svea_"):           return "svea_hovratt_seg"
    if rest.startswith("bergskollegium_a"):return "bergskollegium_adv_seg"
    if rest.startswith("bergskollegium_r"):return "bergskollegium_rel_seg"
    if rest.startswith("bergskollegium_"): return "bergskollegium_seg"
    if rest.startswith("gota_"):           return "gota_hovratt_seg"
    if rest.startswith("krigshovrattens_"):return "krigshovrattens_seg"
    if rest.startswith("trolldomskommiss"):return "trolldomskommissionen_seg"
    if rest.startswith("jonkopings_"):     return "jonkopings_seg"
    if rest.startswith("hanse_"):          return "hanse_kurrent_xvi"
    return "tundmatu"


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        has_allikas = "allikas" in (reader.fieldnames or [])

    # Lisa allikas veerg kui puudub
    for row in rows:
        if not has_allikas or not row.get("allikas"):
            row["allikas"] = detect_allikas(row["failinimi"])

    # Statistika
    counts = Counter(r["allikas"] for r in rows)
    print(f"Kokku: {len(rows)} rida\n")
    print(f"{'Allikas':<42} {'Read':>6}")
    print("-" * 50)
    for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        mark = ""
        if EXCLUDE and src in EXCLUDE:
            mark = "  [VÄLJA]"
        elif INCLUDE and src not in INCLUDE:
            mark = "  [VÄLJA]"
        print(f"  {src:<40} {cnt:>6}{mark}")

    if STATS_ONLY or not OUT_PATH:
        return

    # Filtreerimine
    filtered = rows
    if INCLUDE:
        filtered = [r for r in filtered if r["allikas"] in INCLUDE]
    if EXCLUDE:
        filtered = [r for r in filtered if r["allikas"] not in EXCLUDE]

    # Tasakaalustamine
    if MAX_PER_SOURCE > 0:
        rng = random.Random(SEED)
        by_source: dict[str, list] = defaultdict(list)
        for r in filtered:
            by_source[r["allikas"]].append(r)
        filtered = []
        for src, src_rows in by_source.items():
            if len(src_rows) > MAX_PER_SOURCE:
                src_rows = rng.sample(src_rows, MAX_PER_SOURCE)
            filtered.extend(src_rows)
        rng.shuffle(filtered)

    out = Path(OUT_PATH)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["failinimi", "transkriptsioon", "allikas"])
        writer.writeheader()
        writer.writerows(filtered)

    after = Counter(r["allikas"] for r in filtered)
    print(f"\nKirjutatud: {len(filtered)} rida → {out}")
    for src, cnt in sorted(after.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {cnt:>6}")


if __name__ == "__main__":
    main()
