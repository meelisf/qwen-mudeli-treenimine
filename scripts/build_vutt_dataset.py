#!/usr/bin/env python3
"""
VUTT andmestiku ettevalmistamine treeninguks

Loeb lähtekataloogist kõik teosed, filtreerib leheküljed staatusega
"Valmis" ja loob data/vutt/metadata.csv + kopeerib pildid.

Lähtekataloog: vaikimisi VUTT backup-snapshot ~/vutt-backups/latest/data
(vt scripts/vutt_backup.py VUTT repos). Muu tee: --raw-dir.

Käivitamine:
  python scripts/build_vutt_dataset.py
  python scripts/build_vutt_dataset.py --stats   # ainult statistika, ei kirjuta
  python scripts/build_vutt_dataset.py --raw-dir /mingi/muu/tee
"""

import os
import sys
import json
import csv
import shutil
import re
from datetime import datetime
from pathlib import Path

from convert_marginalia import clean_markup
from imaging import prepare_image, MAX_PIXELS

# Piltide eelskaleerimine on OPT-IN, sest see on inferentsiga seotud:
# eelskaleeritud andmestikul treenitud mudel eeldab, et ka inferents
# skaleerib fit_to_budget()-iga. Vt SPIKKER.md "Piltide eelskaleerimine".
RESIZE_IMAGES = "--resize" in sys.argv

DRY_RUN   = "--stats" in sys.argv

# Lähteandmed. Vaikimisi VUTT backup-snapshot, varuvariandina vana
# vutt_sync.py tõmmis data/vutt-raw/ (kui see veel eksisteerib).
#
# Miks snapshot: VUTT serverist käib üks tõmme (backup) kahe asemel, ja
# andmestik on seotud kindla kuupäevastatud snapshot'iga → reprodutseeritav
# ("treenitud snapshot'ist 20260804T143956Z", vt data/vutt/SOURCE.txt).
#
# NB: snapshot'i sisse EI TOHI kirjutada. Failid on hardlinkidega jagatud
# varasemate snapshot'idega — in-place muudatus muudaks neid kõiki korraga.
# See skript ainult loeb RAW_DIR-ist ja kirjutab OUT_DIR-i.
DEFAULT_RAW_DIRS = [
    Path("~/vutt-backups/latest/data").expanduser(),
    Path("data/vutt-raw"),
]
RAW_DIR = next((p for p in DEFAULT_RAW_DIRS if p.exists()), DEFAULT_RAW_DIRS[0])
for _i, _a in enumerate(sys.argv):
    if _a == "--raw-dir" and _i + 1 < len(sys.argv):
        RAW_DIR = Path(sys.argv[_i + 1]).expanduser()
    elif _a.startswith("--raw-dir="):
        RAW_DIR = Path(_a.split("=", 1)[1]).expanduser()

# `latest` on symlink uusimale snapshot'ile. Lahendame selle KORRA siin, et
# andmestik oleks seotud ühe kindla snapshot'iga ka siis, kui backup vahepeal
# lõpetab ja symlink liigub.
if RAW_DIR.exists():
    RAW_DIR = RAW_DIR.resolve()

OUT_DIR   = Path("data/vutt")
IMG_DIR   = OUT_DIR / "images"
CSV_PATH  = OUT_DIR / "metadata.csv"

VALMIS_STATUSES = {"Valmis"}

# --- Materjali tüüp -------------------------------------------------------
# Trüki- ja käsikirjamudelit treenitakse eraldi, seega andmestik tuleb
# tüübi järgi lahku ajada. VUTT märgib tüübi teose _metadata.json failis
# Wikidata ID-ga: Q1261026 = trükis, Q87167 = käsikiri.
WD_PRINT = "Q1261026"
WD_HAND  = "Q87167"

MATERIAL = "print"          # --type print|hand|all
INCLUDE_UNKNOWN = "--include-unknown" in sys.argv
for _i, _a in enumerate(sys.argv):
    if _a == "--type" and _i + 1 < len(sys.argv):
        MATERIAL = sys.argv[_i + 1]
    elif _a.startswith("--type="):
        MATERIAL = _a.split("=", 1)[1]
if MATERIAL not in ("print", "hand", "all"):
    print(f"Viga: --type peab olema print, hand või all (oli: {MATERIAL})")
    sys.exit(1)


def read_work_type(work_dir: Path) -> str:
    """Tagastab 'print', 'hand' või 'unknown' teose _metadata.json põhjal.

    Väli on ajaloo jooksul olnud mitmes vormis: Wikidata-dict, legacy-dict
    labeliga, ja paljas string. Kõiki tuleb toetada.
    """
    meta = work_dir / "_metadata.json"
    if not meta.exists():
        return "unknown"
    try:
        with open(meta, encoding="utf-8") as f:
            t = json.load(f).get("type")
    except Exception:
        return "unknown"

    if t is None:
        return "unknown"
    if isinstance(t, dict):
        if t.get("id") == WD_PRINT:
            return "print"
        if t.get("id") == WD_HAND:
            return "hand"
        label = (t.get("label") or "").lower()
    elif isinstance(t, str):
        label = t.lower()
    else:
        return "unknown"

    if "käsikiri" in label or "manuscript" in label:
        return "hand"
    if "trükis" in label or "printed" in label:
        return "print"
    return "unknown"


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
        print(f"Viga: {RAW_DIR} puudub.")
        print("  Ootan VUTT backup-snapshot'i: ~/vutt-backups/latest/data")
        print("  Kontrolli, kas öine backup jooksis: journalctl -t vutt-backup --since today")
        print("  Muu allikas: --raw-dir /tee/kataloogile")
        sys.exit(1)

    pairs = []
    skipped_no_json = 0
    skipped_status = 0
    skipped_no_txt = 0
    skipped_empty = 0
    cleaned_markup = 0

    works = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        and d.name != "config"          # VUTT-i seadistuskaust, mitte teos
        and any(d.glob("*.jpg"))        # ilma piltideta kaust pole teos
    )

    print(f"Töötlen {len(works)} teost kataloogist {RAW_DIR}/...")
    print(f"Materjali tüüp: --type {MATERIAL}"
          + ("  (+ tundmatud kaasa)" if INCLUDE_UNKNOWN else ""))

    type_pages = {"print": 0, "hand": 0, "unknown": 0}   # välja jäetud lehed
    unknown_works = []

    for work_dir in works:
        wtype = read_work_type(work_dir)
        if wtype == "unknown" and work_dir not in unknown_works:
            unknown_works.append(work_dir.name)

        keep = (
            MATERIAL == "all"
            or wtype == MATERIAL
            or (wtype == "unknown" and INCLUDE_UNKNOWN)
        )

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

            # Tüübifilter alles siin, et loendur kajastaks päriselt kõlblikke
            # lehti, mitte ka neid, mis oleks niikuinii staatuse tõttu välja
            # kukkunud – muidu näitab statistika petlikult suuri numbreid.
            if not keep:
                type_pages[wtype] += 1
                continue

            cleaned = clean_markup(transcription)
            if cleaned != transcription:
                cleaned_markup += 1
            transcription = cleaned
            if not transcription:
                skipped_empty += 1
                continue

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
    print(f"  Normaliseeritud/puhastatud markup: {cleaned_markup}")
    excluded = {k: v for k, v in type_pages.items() if v}
    if excluded:
        print(f"  Vahele jäetud (vale tüüp): "
              + ", ".join(f"{k}={v}" for k, v in excluded.items()))
    if unknown_works and not INCLUDE_UNKNOWN:
        print(f"\n  NB! {len(unknown_works)} teosel puudub VUTT-is type-väli, "
              f"seega jäid välja ({type_pages['unknown']} Valmis lehte).")
        print(f"      Paranda VUTT-is või kasuta --include-unknown. Teosed:")
        for w in unknown_works[:10]:
            print(f"        {w}")
        if len(unknown_works) > 10:
            print(f"        ... ja veel {len(unknown_works) - 10}")

    if DRY_RUN:
        print("\n--stats: faile ei kirjutata.")
        return

    if not pairs:
        print("\nHoiatus: ühtki sobivat lehekülge ei leitud.")
        return

    # Loo väljundkataloog
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Kopeeri/skaleeri pildid ja kirjuta CSV
    counts = {"resized": 0, "copied": 0, "kept": 0}
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["failinimi", "transkriptsioon"])

        for p in pairs:
            dst = IMG_DIR / p["img_name"]
            if RESIZE_IMAGES:
                counts[prepare_image(p["img_src"], dst)] += 1
            elif not dst.exists():
                shutil.copy2(p["img_src"], dst)
                counts["copied"] += 1
            else:
                counts["kept"] += 1

            writer.writerow([f"images/{p['img_name']}", p["transkriptsioon"]])

    print(f"\nValmis!")
    if RESIZE_IMAGES:
        print(f"  Pildid: skaleeritud {counts['resized']}, "
              f"kopeeritud {counts['copied']}, juba korras {counts['kept']}")
        print(f"  Eelarve: {MAX_PIXELS:,} px (~{MAX_PIXELS // 1024} visuaaltokenit)")
        print()
        print("  !! --resize: see andmestik on EELSKALEERITUD.")
        print("     Sellel treenitud mudel eeldab, et ka inferents kutsub")
        print("     imaging.fit_to_budget(). Lülita kataloogi-jalgimine-ja-ocr.py")
        print("     ja test_model.py ümber SAMAL AJAL kui uue mudeli aktiveerid,")
        print("     muidu tekib treening/inferents-nihe.")
    else:
        print(f"  Pildid kopeeritud: {counts['copied']} "
              f"(olemas juba: {counts['kept']})")
        print(f"  Täissuuruses – protsessor skaleerib treeningu ajal.")
        print(f"  Kiirem torujuhe: --resize (vt SPIKKER.md)")
    print(f"  CSV: {CSV_PATH} ({len(pairs)} rida)")

    # Päritolu: uus mudel tehakse ~korra kuus, seega neli nädalat hiljem ei mäleta
    # keegi, millise andmeseisu pealt see treeniti. Kolm rida, mis selle vastavad.
    source_path = OUT_DIR / "SOURCE.txt"
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(f"raw_dir: {RAW_DIR}\n")
        f.write(f"ehitatud: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"lehti: {len(pairs)}  (--type {MATERIAL}"
                + (", + tundmatud" if INCLUDE_UNKNOWN else "")
                + (", --resize" if RESIZE_IMAGES else "") + ")\n")
    print(f"  Päritolu: {source_path}")

    print(f"\nJärgmine samm: python scripts/train_markup.py [--test]")


if __name__ == "__main__":
    main()
