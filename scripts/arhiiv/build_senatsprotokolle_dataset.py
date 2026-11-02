#!/usr/bin/env python3
"""
Tübingeni Senatsprotokolle (1799-1847) andmestiku töötlemine.

Loeb paarid: Senatsprotokolle/UAT_*/UAT_*_NNN.jpg + page/UAT_*_NNN.xml
Lisab data/kurrent/metadata.csv-le (--append režiimis).

Käivitamine:
  python scripts/build_senatsprotokolle_dataset.py --src /tmp/ubtue-ground-truth/Senatsprotokolle
  python scripts/build_senatsprotokolle_dataset.py --src /tmp/ubtue-ground-truth/Senatsprotokolle --append
"""

import csv
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC     = Path(sys.argv[sys.argv.index("--src") + 1]) if "--src" in sys.argv else Path("/tmp/ubtue-ground-truth/Senatsprotokolle")
APPEND  = "--append" in sys.argv
OUT_DIR = Path("data/kurrent")
IMG_DIR = OUT_DIR / "images"
CSV_PATH = OUT_DIR / "metadata.csv"
MIN_LINES = 3
ALLIKAS = "senatsprotokolle"

PAGEXML_NS = [
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
]


def detect_ns(xml_str):
    for ns in PAGEXML_NS:
        if ns in xml_str:
            return f"{{{ns}}}"
    m = re.search(r'xmlns=["\']([^"\']+pagecontent[^"\']*)["\']', xml_str)
    if m:
        return f"{{{m.group(1)}}}"
    return ""


def parse_pagexml(xml_str):
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    ns = detect_ns(xml_str)
    page = root.find(f"{ns}Page")
    if page is None:
        page = root.find("Page")
        ns = ""
    if page is None:
        return None

    region_order = []
    ro = page.find(f"{ns}ReadingOrder")
    if ro is not None:
        for ref in ro.iter(f"{ns}RegionRefIndexed"):
            region_order.append(ref.get("regionRef"))

    regions = {r.get("id"): r for r in page.findall(f"{ns}TextRegion")}
    ordered = [regions[rid] for rid in region_order if rid in regions]
    for rid, r in regions.items():
        if rid not in region_order:
            ordered.append(r)

    lines = []
    for region in ordered:
        for line in region.findall(f"{ns}TextLine"):
            te = line.find(f"{ns}TextEquiv")
            if te is None:
                continue
            uc = te.find(f"{ns}Unicode")
            if uc is None or not uc.text:
                continue
            text = uc.text.strip()
            if text:
                lines.append(text)

    return "\n".join(lines) if lines else None


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(IMG_DIR.glob("*.jpg"))) if APPEND else 0
    counter = existing

    total = skipped = 0

    csv_mode = "a" if APPEND else "w"
    with open(CSV_PATH, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not APPEND:
            writer.writerow(["failinimi", "transkriptsioon", "allikas"])

        for vol_dir in sorted(SRC.iterdir()):
            if not vol_dir.is_dir():
                continue
            page_dir = vol_dir / "page"
            if not page_dir.exists():
                continue

            for jpg in sorted(vol_dir.glob("*.jpg")):
                xml_path = page_dir / (jpg.stem + ".xml")
                if not xml_path.exists():
                    skipped += 1
                    continue

                xml_str = xml_path.read_text(encoding="utf-8")
                text = parse_pagexml(xml_str)
                if text is None:
                    skipped += 1
                    continue

                line_count = sum(1 for l in text.split("\n") if l.strip())
                if line_count < MIN_LINES:
                    skipped += 1
                    continue

                safe_name = f"{counter:05d}_senatsp_{jpg.stem}.jpg"
                dest = IMG_DIR / safe_name
                shutil.copy2(jpg, dest)
                counter += 1

                writer.writerow([f"images/{safe_name}", text, ALLIKAS])
                total += 1

    print(f"Lisatud: {total} lehekülge  |  vahele jäetud: {skipped}")


if __name__ == "__main__":
    main()
