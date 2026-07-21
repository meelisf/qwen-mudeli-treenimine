#!/usr/bin/env python3
"""
VUTT andmete sünkroniseerimine

Tõmbab rsync-iga VUTT serveri ~/VUTT/data/ sisu siia masinasse
kataloogi data/vutt-raw/.

Käivitamine:
  python scripts/vutt_sync.py          # täisrsync
  python scripts/vutt_sync.py --dry    # näitab mis muutuks, ei tõmba
"""

import os
import sys
import subprocess

DRY_RUN   = "--dry" in sys.argv
VUTT_HOST = "vutt"
VUTT_USER = "meelisf"
VUTT_PATH = "~/VUTT/data/"
LOCAL_RAW = "data/vutt-raw"

def main():
    os.makedirs(LOCAL_RAW, exist_ok=True)

    cmd = [
        "rsync", "-avz", "--progress",
        "--exclude=prosopography/",  # isikute fotod, ei ole OCR materjal
        "--exclude=._trash/",
        "--exclude=_thumbs/",
        "--exclude=.git/",
        "--exclude=uploads/",
        "--include=*/",          # kausta struktuur
        "--include=*.jpg",       # mis saab png-dest jne?
        "--include=*.txt",
        "--include=*.json",
        "--exclude=*",           # kõik muu välja
        f"{VUTT_USER}@{VUTT_HOST}:{VUTT_PATH}",
        LOCAL_RAW + "/",
    ]

    if DRY_RUN:
        cmd.insert(1, "--dry-run")
        print("*** KUIVKÄITUS – muudatusi ei tehta ***")

    print(f"Rsync: {VUTT_USER}@{VUTT_HOST}:{VUTT_PATH} → {LOCAL_RAW}/")
    print("Käsk:", " ".join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nViga: rsync tagastas koodi {result.returncode}")
        sys.exit(1)

    # Statistika
    works = [
        d for d in os.scandir(LOCAL_RAW)
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]
    total_jpg = sum(
        1 for w in works
        for f in os.scandir(w.path)
        if f.name.endswith(".jpg") and not f.name.startswith("_")
    )
    print(f"\nKohalikud andmed ({LOCAL_RAW}/):")
    print(f"  Teoseid:    {len(works)}")
    print(f"  JPG faile:  {total_jpg}")
    print("\nJärgmine samm: python scripts/build_vutt_dataset.py")

if __name__ == "__main__":
    main()
