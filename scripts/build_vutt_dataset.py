#!/usr/bin/env python3
"""
VUTT andmestiku ettevalmistamine treeninguks

Loeb data/vutt-raw/ kataloogist kõik teosed, filtreerib leheküljed
staatusega "Valmis" ja loob data/vutt/metadata.csv + kopeerib pildid.

Käivitamine:
  python scripts/build_vutt_dataset.py
  python scripts/build_vutt_dataset.py --stats   # ainult statistika, ei kirjuta
"""

import os
import sys
import json
import csv
import shutil
import re
from pathlib import Path

from convert_marginalia import convert as convert_marginalia

DRY_RUN   = "--stats" in sys.argv
RAW_DIR   = Path("data/vutt-raw")
OUT_DIR   = Path("data/vutt")
IMG_DIR   = OUT_DIR / "images"
CSV_PATH  = OUT_DIR / "metadata.csv"

VALMIS_STATUSES = {"Valmis"}


def read_page_status(json_path: Path) -> str | None:
    """Loeb lehekülge .json failist staatuse."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status", "")
    except Exception:
        return None


def read_transcription(txt_path: Path) -> str | None:
    """Loeb transkriptsiooni .txt failist."""
    try:
        with open(txt_path, encoding="utf-8") as f:
            text = f.read().strip()
        return text if text else None
    except Exception:
        return None


def safe_image_name(work_name: str, base_name: str) -> str:
    """Loob unikaalse failinime: teos_lehekylg.jpg"""
    # Eemaldame erimärgid failinimest
    work_clean = re.sub(r"[^\w\-]", "_", work_name)[:60]
    return f"{work_clean}__{base_name}"


def main():
    if not RAW_DIR.exists():
        print(f"Viga: {RAW_DIR} puudub. Käivita esmalt: python scripts/vutt_sync.py")
        sys.exit(1)

    pairs = []
    skipped_no_json = 0
    skipped_status = 0
    skipped_no_txt = 0
    skipped_empty = 0

    works = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )

    print(f"Töötlen {len(works)} teost kataloogist {RAW_DIR}/...")

    for work_dir in works:
        jpg_files = sorted(
            f for f in work_dir.iterdir()
            if f.suffix.lower() == ".jpg" and not f.name.startswith("_")
        )

        for jpg_path in jpg_files:
            base = jpg_path.stem  # nt "scan_001"

            # Kontrolli JSON olemasolu ja staatust
            json_path = work_dir / (base + ".json")
            if not json_path.exists():
                skipped_no_json += 1
                continue

            status = read_page_status(json_path)
            if status not in VALMIS_STATUSES:
                skipped_status += 1
                continue

            # Loe transkriptsioon
            txt_path = work_dir / (base + ".txt")
            if not txt_path.exists():
                skipped_no_txt += 1
                continue

            transcription = read_transcription(txt_path)
            if not transcription:
                skipped_empty += 1
                continue

            transcription = convert_marginalia(transcription)

            # Unikaalne pildinimi
            img_name = safe_image_name(work_dir.name, jpg_path.name)
            pairs.append({
                "img_src": jpg_path,
                "img_name": img_name,
                "transkriptsioon": transcription,
                "work": work_dir.name,
            })

    print(f"\nStatistika:")
    print(f"  Leitud Valmis lehekülgi:   {len(pairs)}")
    print(f"  Vahele jäetud (ei JSON):   {skipped_no_json}")
    print(f"  Vahele jäetud (staatus):   {skipped_status}")
    print(f"  Vahele jäetud (ei TXT):    {skipped_no_txt}")
    print(f"  Vahele jäetud (tühi tekst):{skipped_empty}")

    if DRY_RUN:
        print("\n--stats: faile ei kirjutata.")
        return

    if not pairs:
        print("\nHoiatus: ühtki sobivat lehekülge ei leitud.")
        return

    # Loo väljundkataloog
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Kopeeri pildid ja kirjuta CSV
    copied = 0
    skipped_exists = 0
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["failinimi", "transkriptsioon"])

        for p in pairs:
            dst = IMG_DIR / p["img_name"]
            if not dst.exists():
                shutil.copy2(p["img_src"], dst)
                copied += 1
            else:
                skipped_exists += 1

            writer.writerow([f"images/{p['img_name']}", p["transkriptsioon"]])

    print(f"\nValmis!")
    print(f"  Pildid kopeeritud: {copied} (olemas juba: {skipped_exists})")
    print(f"  CSV: {CSV_PATH} ({len(pairs)} rida)")
    print(f"\nJärgmine samm: python scripts/train_markup.py [--test]")


if __name__ == "__main__":
    main()
