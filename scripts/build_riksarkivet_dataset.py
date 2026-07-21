#!/usr/bin/env python3
"""
Riksarkiveti _seg datasettide allalaadimine ja konverteerimine treeninguformaati.

Laadib HuggingFace'ist tar.gz failid (pildid + page_xmls), seob need
failinime järgi, parsib PAGE XML transkriptsioonid, salvestab data/kurrent/.

Käivitamine:
  python scripts/build_riksarkivet_dataset.py
  python scripts/build_riksarkivet_dataset.py --dataset gota_hovratt_seg
  python scripts/build_riksarkivet_dataset.py --dataset krigshovrattens_dombocker_seg --target 3000

Kõik 6 uut datasetti korraga:
  for ds in bergskollegium_advokatfiskalskontoret_seg bergskollegium_relationer_och_skrivelser_seg \
            frihetstidens_utskottshandlingar_seg gota_hovratt_seg \
            krigshovrattens_dombocker_seg trolldomskommissionen_seg; do
    python scripts/build_riksarkivet_dataset.py --dataset $ds
  done
"""

import csv
import io
import os
import re
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import requests
from huggingface_hub import get_token
from PIL import Image
from tqdm import tqdm

# --- Argumendid ---
def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

DATASET    = _arg("--dataset", "gota_hovratt_seg").removeprefix("Riksarkivet/")
TARGET     = int(_arg("--target", 99999))
MIN_LINES  = 5

OUT_DIR    = Path("data/kurrent")
IMG_DIR    = OUT_DIR / "images"
CSV_PATH   = OUT_DIR / "metadata.csv"

# Loe .env failist kui env muutuja pole seatud
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("HF_TOKEN=") and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = _line.split("=", 1)[1].strip()

# Kasuta env tokenit või huggingface_hub'i salvestatud tokenit
HF_TOKEN   = os.environ.get("HF_TOKEN") or get_token() or ""

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
    page = root.find(f"{ns}Page") or root.find("Page")
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


def hf_url(dataset: str, path: str) -> str:
    return f"https://huggingface.co/datasets/Riksarkivet/{dataset}/resolve/main/{path}"


def download_file(url: str, dest: Path) -> None:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    print(f"  Laadin: {url}")
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def detect_num_parts(dataset: str) -> int:
    """Küsib HF API-st mitu tar.gz osa on."""
    url = f"https://huggingface.co/api/datasets/Riksarkivet/{dataset}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    r = requests.get(url, headers=headers)
    if not r.ok:
        return 2  # vaikimisi
    siblings = r.json().get("siblings", [])
    imgs = [s["rfilename"] for s in siblings
            if re.match(r"data/images/.*_images_\d+\.tar\.gz", s["rfilename"])]
    return len(imgs) if imgs else 2


def process_pair(images_tar: Path, xmls_tar: Path, dataset_short: str,
                 counter: int, total_so_far: int) -> tuple[int, int, dict]:
    stats = defaultdict(int)
    added = 0

    print(f"  Loeme XML-id: {xmls_tar.name}")
    xml_by_stem: dict[str, str] = {}
    with tarfile.open(xmls_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".xml"):
                continue
            stem = Path(member.name).stem
            f = tf.extractfile(member)
            if f:
                xml_by_stem[stem] = f.read().decode("utf-8", errors="replace")
    print(f"    {len(xml_by_stem)} XML faili")

    print(f"  Töötleme pilte: {images_tar.name}")
    with tarfile.open(images_tar, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if m.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        print(f"    {len(members)} pilti")

        # Kontrolli kas CSV-l on allikas veerg
        has_allikas = False
        if CSV_PATH.exists():
            with open(CSV_PATH, newline="", encoding="utf-8") as f:
                has_allikas = "allikas" in (next(csv.reader(f), []))

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)

            for member in tqdm(members, desc=f"  {dataset_short}", unit="lk"):
                if total_so_far + added >= TARGET:
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

                if sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES:
                    stats["liiga_lühike"] += 1
                    continue

                try:
                    img_bytes = tf.extractfile(member).read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    stats["pildi_viga"] += 1
                    continue

                safe_stem = re.sub(r"[^\w\-]", "_", stem)[:60]
                safe_name = f"{counter:05d}_{dataset_short}_{safe_stem}.jpg"
                img.save(IMG_DIR / safe_name, "JPEG", quality=90)

                if has_allikas:
                    writer.writerow([f"images/{safe_name}", text, DATASET])
                else:
                    writer.writerow([f"images/{safe_name}", text])
                counter += 1
                added += 1

            csvf.flush()

    return counter, added, dict(stats)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        print("HOIATUS: HF_TOKEN puudub – privaatsete datasettide allalaadimine võib ebaõnnestuda")
    else:
        print(f"HF token: OK ({HF_TOKEN[:8]}...)")

    num_parts = detect_num_parts(DATASET)
    short = re.sub(r"[^\w]", "_", DATASET)[:16]
    print(f"\nDataset: Riksarkivet/{DATASET}  ({num_parts} osa)")

    counter = len(list(IMG_DIR.glob("*.jpg")))
    total_added = 0

    with tempfile.TemporaryDirectory(prefix="riksark_") as tmpdir:
        tmp = Path(tmpdir)

        for i in range(1, num_parts + 1):
            if total_added >= TARGET:
                break

            img_path = tmp / f"{DATASET}_images_{i}.tar.gz"
            xml_path = tmp / f"{DATASET}_page_xmls_{i}.tar.gz"

            print(f"\nOsa {i}/{num_parts}:")
            download_file(hf_url(DATASET, f"data/images/{DATASET}_images_{i}.tar.gz"), img_path)
            download_file(hf_url(DATASET, f"data/page_xmls/{DATASET}_page_xmls_{i}.tar.gz"), xml_path)

            counter, added, stats = process_pair(img_path, xml_path, short, counter, total_added)
            total_added += added

            print(f"  Lisatud: {added} (kokku selles datasetis: {total_added})")
            for k, v in sorted(stats.items()):
                print(f"    Vahele ({k}): {v}")

    print(f"\nValmis: {total_added} lehekülge datasetist '{DATASET}'")


if __name__ == "__main__":
    main()
    os._exit(0)
