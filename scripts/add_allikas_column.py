#!/usr/bin/env python3
"""
Üks kord jooksev migratsiooniskript: lisab metadata.csv-le 'allikas' veeru.

Allikas tuletakse failinimest. Käivita üks kord, siis kirjutab
faili üle kolme veeruga (failinimi, transkriptsioon, allikas).

Käivitamine:
  python scripts/add_allikas_column.py
  python scripts/add_allikas_column.py --dry-run   # ainult näitab, ei kirjuta
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path("data/kurrent/metadata.csv")
DRY_RUN  = "--dry-run" in sys.argv


def detect_allikas(failinimi: str) -> str:
    name = Path(failinimi).name  # nt "00000_083605_0307_3473452.jpg"
    # Eemalda järjekorranumber (5 numbrit + _)
    rest = re.sub(r"^\d{5}_", "", name)

    if re.match(r"^08\d{4}_", rest):
        return "kurrent_xix"
    if rest.startswith("aaeb_"):
        return "aaeb_xiv_xvii"
    if rest.startswith("bullinger_"):
        return "bullinger_autoren"
    if rest.startswith("koenigsfelde"):
        return "koenigsfelden_adhr"
    if rest.startswith("svea_"):
        return "svea_hovratt_seg"
    if rest.startswith("bergskollegium_a"):
        return "bergskollegium_adv_seg"
    if rest.startswith("bergskollegium_r"):
        return "bergskollegium_rel_seg"
    if rest.startswith("bergskollegium_"):
        return "bergskollegium_seg"
    if rest.startswith("gota_"):
        return "gota_hovratt_seg"
    if rest.startswith("krigshovrattens_"):
        return "krigshovrattens_seg"
    if rest.startswith("trolldomskommiss"):
        return "trolldomskommissionen_seg"
    if rest.startswith("jonkopings_") or rest.startswith("jonk_"):
        return "jonkopings_seg"
    if rest.startswith("hanse_"):
        return "hanse_kurrent_xvi"
    return "tundmatu"


def main():
    if not CSV_PATH.exists():
        print(f"Viga: {CSV_PATH} ei eksisteeri")
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "allikas" in header:
        print(f"'allikas' veerg on juba olemas ({len(rows)} rida). Täidan tühjad.")
        updated = 0
        for row in rows:
            if not row.get("allikas"):
                row["allikas"] = detect_allikas(row["failinimi"])
                updated += 1
        print(f"  Täidetud: {updated} rida")
    else:
        print(f"Lisan 'allikas' veeru {len(rows)} reale ...")
        for row in rows:
            row["allikas"] = detect_allikas(row["failinimi"])

    counts = Counter(r["allikas"] for r in rows)
    print("\nAllikate jaotus:")
    for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {src:40s} {cnt:6d}")

    if DRY_RUN:
        print("\n--dry-run: faili ei kirjutata")
        return

    tmp = CSV_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["failinimi", "transkriptsioon", "allikas"])
        writer.writeheader()
        writer.writerows(rows)

    CSV_PATH.replace(CSV_PATH.with_suffix(".bak"))
    tmp.rename(CSV_PATH)
    print(f"\nValmis. Varukoopia: {CSV_PATH.with_suffix('.bak')}")


if __name__ == "__main__":
    main()
