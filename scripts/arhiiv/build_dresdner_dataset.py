#!/usr/bin/env python3
"""
Dresdner Hofdiarium 1673 allalaadimine Zenodost ja konverteerimine treeninguformaati.

20 täislehekülge XVII sajandi saksoni Kanzleikurrentit (SLUB Mscr.Dresd.K.117).
Formaadis: JPG pildid + ALTO XML (v4) transkriptsioonid.
Litsents: CC BY-NC-SA 4.0
DOI: 10.5281/zenodo.15303243

Käivitamine:
  python scripts/build_dresdner_dataset.py
"""

import csv
import io
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

OUT_DIR  = Path("data/kurrent")
IMG_DIR  = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "metadata.csv"
ALLIKAS  = "dresdner_hofdiarium_1673"

ZENODO_BASE = "https://zenodo.org/records/15303243/files"
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"

MIN_LINES = 3


def fetch(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_alto(xml_bytes: bytes) -> str | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    ns = f"{{{ALTO_NS}}}"
    # Kogu kõik TextLine-d, sorteeri VPOS järgi
    lines_data = []
    for block in root.iter(f"{ns}TextBlock"):
        for tl in block.iter(f"{ns}TextLine"):
            vpos = float(tl.get("VPOS", 0))
            # Kogu String CONTENT atribuudid (tavaliselt üks rida = üks String)
            parts = [s.get("CONTENT", "") for s in tl.iter(f"{ns}String")
                     if s.get("CONTENT", "").strip()]
            if parts:
                lines_data.append((vpos, " ".join(parts)))

    lines_data.sort(key=lambda x: x[0])
    lines = [text for _, text in lines_data]
    return "\n".join(lines) if lines else None


def get_file_pairs() -> list[tuple[str, str]]:
    """Tagastab (img_filename, xml_filename) paare Zenodo API kaudu."""
    r = requests.get("https://zenodo.org/api/records/15303243", timeout=30)
    r.raise_for_status()
    files = [f["key"] for f in r.json()["files"]]
    jpgs = {Path(f).stem: f for f in files if f.endswith(".jpg")}
    xmls = {Path(f).stem: f for f in files if f.endswith(".xml")}
    pairs = []
    for stem in sorted(jpgs):
        if stem in xmls:
            pairs.append((jpgs[stem], xmls[stem]))
    return pairs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    print("Zenodo API: otsib faile ...")
    pairs = get_file_pairs()
    print(f"  {len(pairs)} lehekülge")

    counter = len(list(IMG_DIR.glob("*.jpg")))
    added = 0
    stats = defaultdict(int)

    has_allikas = False
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            has_allikas = "allikas" in (next(csv.reader(f), []))

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)

        for img_fn, xml_fn in tqdm(pairs, desc="Töötlus", unit="lk"):
            print(f"  {img_fn}")

            xml_bytes = fetch(f"{ZENODO_BASE}/{xml_fn}")
            text = parse_alto(xml_bytes)
            if text is None:
                stats["xml_viga"] += 1
                continue
            if sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES:
                stats["liiga_lühike"] += 1
                continue

            img_bytes = fetch(f"{ZENODO_BASE}/{img_fn}")
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                stats["pildi_viga"] += 1
                continue

            stem = re.sub(r"[^\w\-]", "_", Path(img_fn).stem)[:60]
            safe_name = f"{counter:05d}_dresdner_{stem}.jpg"
            img.save(IMG_DIR / safe_name, "JPEG", quality=92)

            if has_allikas:
                writer.writerow([f"images/{safe_name}", text, ALLIKAS])
            else:
                writer.writerow([f"images/{safe_name}", text])

            counter += 1
            added += 1

        csvf.flush()

    print(f"\nValmis: {added} lehekülge ({ALLIKAS})")
    for k, v in sorted(stats.items()):
        print(f"  Vahele ({k}): {v}")


if __name__ == "__main__":
    main()
    os._exit(0)
