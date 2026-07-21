#!/usr/bin/env python3
"""
fgho/hanse-kurrent-xvi-rawxml allalaadimine ja konverteerimine treeninguformaati.

1145 lehekülge XVI sajandi Hansaliidu Kurrent käsikirja (Lübeck, Stralsund, Köln jne).
Parquet formaadis: image (PIL) + xml_content (PAGE XML string).

Käivitamine:
  python scripts/build_hanse_dataset.py
  python scripts/build_hanse_dataset.py --target 500
"""

import csv
import io
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image
from tqdm import tqdm

# --- Argumendid ---
def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

TARGET     = int(_arg("--target", 99999))
MIN_LINES  = 3  # Hanse dokumendid on tihti lühemad

OUT_DIR    = Path("data/kurrent")
IMG_DIR    = OUT_DIR / "images"
CSV_PATH   = OUT_DIR / "metadata.csv"
ALLIKAS    = "hanse_kurrent_xvi"

# Loe token .env failist või huggingface_hub cache'ist
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("HF_TOKEN=") and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = _line.split("=", 1)[1].strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token() or ""

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

    if HF_TOKEN:
        print(f"HF token: OK ({HF_TOKEN[:8]}...)")
    else:
        print("HOIATUS: HF_TOKEN puudub")

    print("Laadin fgho/hanse-kurrent-xvi-rawxml ...")
    ds = load_dataset("fgho/hanse-kurrent-xvi-rawxml", split="train",
                      token=HF_TOKEN or None)
    print(f"  {len(ds)} näidet")

    counter = len(list(IMG_DIR.glob("*.jpg")))
    added = 0
    stats = defaultdict(int)

    # Kontrolli kas CSV-l on juba allikas veerg
    has_allikas = False
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
            has_allikas = "allikas" in header

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)

        for row in tqdm(ds, desc="Töötlus", unit="lk"):
            if added >= TARGET:
                break

            xml_str = row.get("xml_content", "")
            if not xml_str:
                stats["xml_puudub"] += 1
                continue

            text = parse_pagexml(xml_str)
            if text is None:
                stats["xml_viga"] += 1
                continue

            if sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES:
                stats["liiga_lühike"] += 1
                continue

            img_raw = row.get("image")
            if img_raw is None:
                stats["pilt_puudub"] += 1
                continue

            try:
                if isinstance(img_raw, Image.Image):
                    img = img_raw.convert("RGB")
                elif isinstance(img_raw, dict) and "bytes" in img_raw:
                    img = Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
                elif isinstance(img_raw, bytes):
                    img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                else:
                    stats["pildi_formaat"] += 1
                    continue
            except Exception:
                stats["pildi_viga"] += 1
                continue

            stem = re.sub(r"[^\w\-]", "_", row.get("filename", str(counter)))[:60]
            safe_name = f"{counter:05d}_hanse_{stem}.jpg"
            img.save(IMG_DIR / safe_name, "JPEG", quality=90)

            if has_allikas:
                writer.writerow([f"images/{safe_name}", text, ALLIKAS])
            else:
                writer.writerow([f"images/{safe_name}", text])

            counter += 1
            added += 1

        csvf.flush()

    print(f"\nValmis: {added} lehekülge (hanse_kurrent_xvi)")
    for k, v in sorted(stats.items()):
        print(f"  Vahele ({k}): {v}")


if __name__ == "__main__":
    main()
    os._exit(0)
