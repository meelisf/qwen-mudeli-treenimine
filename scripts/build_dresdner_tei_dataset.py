#!/usr/bin/env python3
"""
Dresdeni Hofdiarium 1665 (Zenodo 14932508) andmestiku ehitamine.

TEI XML: lokaalne fail (--xml) või allalaadimine Zenodost.
Pildid: otse SLUB Dresdeni serverist (facs URL-id XML-is).

Käivitamine:
  python scripts/build_dresdner_tei_dataset.py --xml /tmp/hofdiarium.xml --append
"""

import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from PIL import Image
import io

XML_PATH = Path(sys.argv[sys.argv.index("--xml") + 1]) if "--xml" in sys.argv else None
APPEND   = "--append" in sys.argv
OUT_DIR  = Path("data/kurrent")
IMG_DIR  = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "metadata.csv"
MIN_LINES = 3
ALLIKAS  = "dresdner_1665"
ZENODO_XML_URL = "https://zenodo.org/records/14932508/files/SLUB%20Dresd%20Mscr%20K80%20(Hofdiarium%201665)%20Release%202.xml?download=1"

TEI_NS = "http://www.tei-c.org/ns/1.0"


def download_xml() -> bytes:
    print("Laadin XML-i Zenodost...")
    r = requests.get(ZENODO_XML_URL, timeout=60)
    r.raise_for_status()
    return r.content


def collect_pages(root) -> list[tuple[str, str, str]]:
    """
    Tagastab [(facs_url, page_type, text), ...] kõigi pb elementide jaoks.
    Tekst = kõik tekstisisu kuni järgmise pb-ni.
    """
    ns = f"{{{TEI_NS}}}"
    body = root.find(f".//{ns}body")
    if body is None:
        body = root.find(".//body")
        ns = ""

    # Kogu elemendipuu lapikuks listiks
    all_elems = list(body.iter())

    # Leia pb indeksid
    pb_indices = [i for i, e in enumerate(all_elems) if e.tag.split("}")[-1] == "pb" and e.get("facs")]

    pages = []
    for idx, pb_idx in enumerate(pb_indices):
        pb = all_elems[pb_idx]
        facs = pb.get("facs", "")
        ptype = pb.get("type", "")

        # Tekst: serialiseeri segmendi XML stringiks, asenda <lb/> reavahetega
        next_pb = pb_indices[idx + 1] if idx + 1 < len(pb_indices) else len(all_elems)
        segment = all_elems[pb_idx + 1 : next_pb]

        # Kogu tail-tekstid: <lb/> tail on järgmise rea tekst
        # Muude elementide text/tail lisatakse samale reale
        lines = []
        current_line = []

        def add_text(t):
            if t:
                t = t.strip()
                if t:
                    current_line.append(t)

        def flush():
            if current_line:
                lines.append(" ".join(current_line))
                current_line.clear()

        for elem in segment:
            tag = elem.tag.split("}")[-1]
            if tag == "note":
                # Jäta kõrvalmärkused vahele, säilita tail
                add_text(elem.tail)
                continue
            if tag == "lb":
                # lb tail on järgmise rea sisu – reavahe enne seda
                flush()
                add_text(elem.tail)
                continue
            add_text(elem.text)
            add_text(elem.tail)

        flush()
        text = "\n".join(l for l in lines if l)
        pages.append((facs, ptype, text))

    return pages


def download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"  Pildi viga {url}: {e}")
        return None


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    if XML_PATH and XML_PATH.exists():
        xml_bytes = XML_PATH.read_bytes()
    else:
        xml_bytes = download_xml()

    root = ET.fromstring(xml_bytes)
    pages = collect_pages(root)

    diary_pages = [(f, t, txt) for f, t, txt in pages if t == "diaryEntry"]
    print(f"Kokku leheküljed: {len(pages)}  |  diaryEntry: {len(diary_pages)}")

    existing = len(list(IMG_DIR.glob("*.jpg"))) if APPEND else 0
    counter = existing
    total = skipped = 0

    csv_mode = "a" if APPEND else "w"
    with open(CSV_PATH, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not APPEND:
            writer.writerow(["failinimi", "transkriptsioon", "allikas"])

        for facs, ptype, text in diary_pages:
            line_count = sum(1 for l in text.split("\n") if l.strip())
            if line_count < MIN_LINES:
                skipped += 1
                continue

            img = download_image(facs)
            if img is None:
                skipped += 1
                continue

            page_num = re.search(r"(\d+)\.tif", facs)
            stem = page_num.group(1) if page_num else f"{counter:05d}"
            safe_name = f"{counter:05d}_dresdner1665_{stem}.jpg"
            img.save(IMG_DIR / safe_name, "JPEG", quality=90)
            counter += 1

            writer.writerow([f"images/{safe_name}", text, ALLIKAS])
            total += 1

            if total % 10 == 0:
                f.flush()
                print(f"  {total}/{len(diary_pages)} lehekülge allalaetud...")
            time.sleep(0.3)

    print(f"\nLisatud: {total}  |  vahele jäetud: {skipped}")


if __name__ == "__main__":
    main()
