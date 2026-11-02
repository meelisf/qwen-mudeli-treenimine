#!/usr/bin/env python3
"""
Käsikirjaandmestiku allalaadimine ja konverteerimine treeninguformaati.

Toetab kõiki dh-unibe/image-text_* andmestikke (sama parquet+pagexml formaat).
Laadib streaminguga, filtreerib PageXML-ist teksti,
salvestab data/kurrent/metadata.csv + images/.

Käivitamine:
  python scripts/build_kurrent_dataset.py
  python scripts/build_kurrent_dataset.py --target 2000
  python scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_bullinger-autoren --target 2000 --append
  python scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_aaeb-xiv-xvii --target 2000 --append
  python scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_koenigsfelden-adhr-colmar --append
"""

import csv
import io
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, Image as HFImage
from huggingface_hub import login as hf_login
from PIL import Image
from tqdm import tqdm

# --- Argumendid ---
def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

DATASET    = _arg("--dataset", "dh-unibe/image-text_kurrent-xix")
TARGET     = int(_arg("--target", 8000))
MAX_PER_PROJECT = int(_arg("--max-per-project", 250))
MIN_LINES  = 5
APPEND     = "--append" in sys.argv

OUT_DIR    = Path("data/kurrent")
IMG_DIR    = OUT_DIR / "images"
CSV_PATH   = OUT_DIR / "metadata.csv"

# Andmestiku lühinimi failiprefiksiks
_DS_SHORT  = DATASET.split("/")[-1].replace("image-text_", "")[:12].replace("-", "_")

# PageXML namespace variandid
PAGEXML_NS = [
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
]


def detect_ns(xml_str: str) -> str:
    for ns in PAGEXML_NS:
        if ns in xml_str:
            return f"{{{ns}}}"
    # Proovi regex-iga
    m = re.search(r'xmlns=["\']([^"\']+pagecontent[^"\']*)["\']', xml_str)
    if m:
        return f"{{{m.group(1)}}}"
    return ""


def parse_pagexml(xml_str: str) -> str | None:
    """Tagastab transkriptsiooni stringi või None kui pole kasutatav."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    ns = detect_ns(xml_str)
    page = root.find(f"{ns}Page")
    if page is None:
        ns = ""
        page = root.find("Page")
    if page is None:
        return None

    # Loe lugemiskord
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


def load_pil(img_field) -> Image.Image | None:
    """Konverteerib HuggingFace image välja PIL Image-ks."""
    try:
        if isinstance(img_field, Image.Image):
            return img_field
        if isinstance(img_field, dict):
            raw = img_field.get("bytes") or img_field.get("path")
            if isinstance(raw, bytes):
                return Image.open(io.BytesIO(raw)).convert("RGB")
        # Viimane varuvariants
        return Image.open(io.BytesIO(bytes(img_field))).convert("RGB")
    except Exception:
        return None


def main():
    # HF autentimine – tokenist env muutuja kaudu
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    # Append-režiimis: alusta loendurit olemasolevast mahust
    existing = len(list(IMG_DIR.glob("*.jpg"))) if APPEND else 0
    counter = existing

    project_counts: dict[str, int] = defaultdict(int)
    total = 0
    stats = defaultdict(int)

    print(f"Andmestik: {DATASET}  target={TARGET}  append={APPEND}  olemasolev={existing}")

    dataset = load_dataset(
        DATASET,
        split="train",
        streaming=True,
    ).cast_column("image", HFImage(decode=False))

    csv_mode = "a" if APPEND else "w"
    with open(CSV_PATH, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not APPEND:
            writer.writerow(["failinimi", "transkriptsioon"])

        with tqdm(total=TARGET, desc="Laadimine", unit="lk") as pbar:
            for row in dataset:
                if total >= TARGET:
                    break

                project = row["project_name"]
                filename = row["filename"]

                if project_counts[project] >= MAX_PER_PROJECT:
                    stats["projekt_täis"] += 1
                    continue

                text = parse_pagexml(row["xml_content"])

                if text is None:
                    stats["xml_viga"] += 1
                    continue

                line_count = sum(1 for l in text.split("\n") if l.strip())
                if line_count < MIN_LINES:
                    stats["liiga_lühike"] += 1
                    continue

                img = load_pil(row["image"])
                if img is None:
                    stats["pildi_viga"] += 1
                    continue

                # Unikaalne failinimi: counter_ds_originaal.jpg
                stem = re.sub(r"[^\w\-]", "_", Path(filename).stem)
                safe_name = f"{counter:05d}_{_DS_SHORT}_{stem}.jpg"
                counter += 1
                img_path = IMG_DIR / safe_name
                img.save(img_path, "JPEG", quality=90)

                writer.writerow([f"images/{safe_name}", text])
                project_counts[project] += 1
                total += 1
                pbar.update(1)

                if total % 500 == 0:
                    f.flush()

    print(f"\nKogutud: {total} lehekülge ({DATASET})")
    print(f"Projektide arv: {len(project_counts)}")
    for k, v in sorted(stats.items()):
        print(f"  Vahele jäetud ({k}): {v}")
    print(f"\nTop 10 projekti:")
    for p, c in sorted(project_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {c:4d}  {p}")


if __name__ == "__main__":
    main()
    os._exit(0)  # Väldib streaming thread cleanup crash-i
