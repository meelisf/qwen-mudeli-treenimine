#!/usr/bin/env python3
"""
Rootsi Svea hovrätt käsikirjaandmestiku konverteerimine treeninguformaati.

Loeb lokaalseid tar.gz faile (pildid + pagexml), seob need failinime järgi,
parsib transkriptsioonid ja lisab data/kurrent/ andmestikku.

Käivitamine:
  python scripts/build_svea_dataset.py
  python scripts/build_svea_dataset.py --images data/raw/svea_hovratt_seg_images_1.tar.gz \
                                       --xmls   data/raw/svea_hovratt_seg_page_xmls_1.tar.gz
"""

import csv
import io
import os
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# --- Argumendid ---
def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

IMAGES_TAR = Path(_arg("--images", "data/raw/svea_hovratt_seg_images_1.tar.gz"))
XMLS_TAR   = Path(_arg("--xmls",   "data/raw/svea_hovratt_seg_page_xmls_1.tar.gz"))
TARGET     = int(_arg("--target", 99999))
MIN_LINES  = 5

OUT_DIR    = Path("data/kurrent")
IMG_DIR    = OUT_DIR / "images"
CSV_PATH   = OUT_DIR / "metadata.csv"

PAGEXML_NS = [
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
]


def detect_ns(xml_str: str) -> str:
    for ns in PAGEXML_NS:
        if ns in xml_str:
            return f"{{{ns}}}"
    m = re.search(r'xmlns=["\']([^"\']+pagecontent[^"\']*)["\']', xml_str)
    if m:
        return f"{{{m.group(1)}}}"
    return ""


def parse_pagexml(xml_str: str) -> str | None:
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    ns = detect_ns(xml_str)
    page = root.find(f"{ns}Page")
    if page is None:
        page = root.find("Page")
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    # Loeme kõik XML-id mällu (7MB, mahub)
    print(f"Loeme XML-id: {XMLS_TAR}")
    xml_by_stem: dict[str, str] = {}
    with tarfile.open(XMLS_TAR, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".xml"):
                continue
            stem = Path(member.name).stem
            f = tf.extractfile(member)
            if f:
                xml_by_stem[stem] = f.read().decode("utf-8", errors="replace")
    print(f"  {len(xml_by_stem)} XML faili laetud")

    # Alusta loendurit olemasolevast mahust
    counter = len(list(IMG_DIR.glob("*.jpg")))
    total = 0
    stats = defaultdict(int)

    print(f"Töötleme pilte: {IMAGES_TAR}  (olemasolev={counter})")

    with tarfile.open(IMAGES_TAR, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        print(f"  {len(members)} pilti arhiivis")

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)

            for member in tqdm(members, desc="Töötlus", unit="lk"):
                if total >= TARGET:
                    break

                stem = Path(member.name).stem
                xml_str = xml_by_stem.get(stem)
                if xml_str is None:
                    stats["xml_puudub"] += 1
                    continue

                text = parse_pagexml(xml_str)
                if text is None:
                    stats["xml_viga"] += 1
                    continue

                line_count = sum(1 for l in text.split("\n") if l.strip())
                if line_count < MIN_LINES:
                    stats["liiga_lühike"] += 1
                    continue

                # Lae pilt
                try:
                    img_bytes = tf.extractfile(member).read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    stats["pildi_viga"] += 1
                    continue

                safe_stem = re.sub(r"[^\w\-]", "_", stem)[:60]
                safe_name = f"{counter:05d}_svea_{safe_stem}.jpg"
                img.save(IMG_DIR / safe_name, "JPEG", quality=90)

                writer.writerow([f"images/{safe_name}", text])
                counter += 1
                total += 1

            csvf.flush()

    print(f"\nKogutud: {total} lehekülge (svea_hovratt)")
    for k, v in sorted(stats.items()):
        print(f"  Vahele jäetud ({k}): {v}")


if __name__ == "__main__":
    main()
    os._exit(0)
