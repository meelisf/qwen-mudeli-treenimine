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

Tühjad leheküljed Kurrent-andmestikku (vt SPIKKER.md "Tühjad leheküljed"):
  python scripts/build_vutt_dataset.py --type hand --only-empty --stats
  python scripts/build_vutt_dataset.py --type hand --only-empty \\
      --out data/kurrent --append --allikas vutt_tyhjad
"""

import os
import sys
import json
import csv
import shutil
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

from convert_marginalia import clean_markup
from imaging import prepare_image, MAX_PIXELS
from lyhend_makron_trukk import puhasta
from prompt import EMPTY_PAGE_MARKER

# Terve lehekülje transkriptsioon võib ületada csv-mooduli vaikimisi
# väljapiirangut (128 kB), kui loeme olemasolevat andmestikku --append jaoks.
csv.field_size_limit(10 ** 7)

# Piltide eelskaleerimine on OPT-IN, sest see on inferentsiga seotud:
# eelskaleeritud andmestikul treenitud mudel eeldab, et ka inferents
# skaleerib fit_to_budget()-iga. Vt SPIKKER.md "Piltide eelskaleerimine".
RESIZE_IMAGES = "--resize" in sys.argv
#: `--keep-m-italics` jätab `<i>` `<m>` sisse alles. Vaikimisi võtab
#: `clean_markup` need maha – aga siis EI SAA `train_markup.py --keep-m-italics`
#: neid enam tagasi tuua, sest CSV-s pole neid enam. Mõõdetud 28.08: strippimine
#: kaotab VUTT-i poolelt 4 851 `<i>`-d (14 527 -> 9 676). Kui ehitad andmestikku
#: `--keep-m-italics` A/B jaoks, PEAB see lipp siin ka olema.
KEEP_M_ITALICS = "--keep-m-italics" in sys.argv

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

DEFAULT_OUT_DIR = Path("data/vutt")
OUT_DIR = DEFAULT_OUT_DIR
OUT_EXPLICIT = False
for _i, _a in enumerate(sys.argv):
    if _a == "--out" and _i + 1 < len(sys.argv):
        OUT_DIR, OUT_EXPLICIT = Path(sys.argv[_i + 1]).expanduser(), True
    elif _a.startswith("--out="):
        OUT_DIR, OUT_EXPLICIT = Path(_a.split("=", 1)[1]).expanduser(), True
IMG_DIR   = OUT_DIR / "images"
CSV_PATH  = OUT_DIR / "metadata.csv"

# --append lisab olemasolevale CSV-le read juurde (olemasolevat sisu ei
# kirjutata üle). Nii saab VUTT-i lehed panna Kurrent-andmestikku, mis on
# 17 000 rida välisallikaid – ülekirjutamine hävitaks selle.
APPEND = "--append" in sys.argv
FORCE  = "--force" in sys.argv

# Kolmanda veeru (`allikas`) väärtus, kui sihtCSV seda kasutab.
ALLIKAS = "vutt"
ALLIKAS_EXPLICIT = False
for _i, _a in enumerate(sys.argv):
    if _a == "--allikas" and _i + 1 < len(sys.argv):
        ALLIKAS, ALLIKAS_EXPLICIT = sys.argv[_i + 1], True
    elif _a.startswith("--allikas="):
        ALLIKAS, ALLIKAS_EXPLICIT = _a.split("=", 1)[1], True

# --only-empty: võta ainult tühjaks märgitud leheküljed. Ilma selleta tõmbaks
# `--out data/kurrent` kaasa kogu VUTT-i käsikirjamaterjali koos XML
# märgendusega, mida Kurrent-mudel ei oska ega taha.
ONLY_EMPTY = "--only-empty" in sys.argv

# --max-chars N: võta ainult hõredad lehed – need, mille tekstis on kuni N
# tähemärki (märgendid maha arvatud). Tühjaks märgitud lehed mahuvad siia
# alati sisse, sest märgend ise on 15 märki.
#
# Miks: mudel on treenitud ainult täistekstiga lehtedel (data/vutt mediaan
# 2254 märki, ALLA 100 märgi mitte ühtegi lehte), seega eeldab ta, et igal
# lehel peab teksti palju olema. Kui lehel on ainult leheküljenumber või
# paar rida, satub ta segadusse ja hakkab juurde luuletama.
MAX_CHARS = None
for _i, _a in enumerate(sys.argv):
    if _a == "--max-chars" and _i + 1 < len(sys.argv):
        MAX_CHARS = int(sys.argv[_i + 1])
    elif _a.startswith("--max-chars="):
        MAX_CHARS = int(_a.split("=", 1)[1])

# Millest allpool loeme lehte "hõredaks" aruandes (ei filtreeri midagi).
SPARSE_LIMIT = MAX_CHARS if MAX_CHARS is not None else 100

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

# data/vutt on TRÜKI markup-andmestik (train_markup.py loeb sealt). Käsikirja-
# jooks ilma --out liputa kirjutaks selle üle ja järgmine markup-treening
# treeniks vaikselt vale materjali peal.
if MATERIAL != "print" and not OUT_EXPLICIT and not DRY_RUN:
    print(f"Viga: --type {MATERIAL} kirjutaks üle {DEFAULT_OUT_DIR}/, "
          f"mis on TRÜKI markup-andmestik.")
    print("  Anna sihtkoht: --out data/kurrent --append --allikas vutt_tyhjad")
    print(f"  Kui tahad päriselt {DEFAULT_OUT_DIR}/ üle kirjutada: "
          f"--out {DEFAULT_OUT_DIR}")
    sys.exit(1)


# --- Tühja lehekülje märgend ---------------------------------------------
# VUTT-is käsitsi märgendades on lihtne kirjutada "tühi leht", "[Tühi
# lehekülg]" või jätta sulud ära. Treeningu jaoks on oluline, et string oleks
# BAIT-BAIT sama – muidu ei õpi mudel seda lõpetamismärgina, mis oli terve
# harjutuse mõte. Seepärast tunneme variandid ära ja kaebame nende üle,
# selle asemel et vaikselt sisse lasta või vaikselt parandada.
EMPTY_VARIANTS = {
    "tühi lehekülg", "tühi lehekülg.", "tühi leht", "tühi lk", "tühi",
    "lehekülg tühi", "leht tühi", "tyhi lehekylg", "tyhi lehekulg",
    "empty page", "blank page", "empty", "blank", "leer", "leere seite",
    "vacat", "vacuum",
}


def plain_length(text: str) -> int:
    """Teksti pikkus ilma XML-märgenditeta ja korduvate tühikuteta.

    `<i>A</i>` on kaks korda pikem kui `A`, aga mudeli jaoks on see üks
    täht – hõreduse mõõtmisel loeb sisu, mitte märgendus.
    """
    plain = re.sub(r"<[^>]+>", "", text)
    return len(re.sub(r"\s+", " ", plain).strip())


def _fold(s: str) -> str:
    """Täpitähed maha – 'tühi lehekülg' ja 'tyhi lehekulg' on sama kirjaviga."""
    return s.translate(str.maketrans("üõöäåÜÕÖÄÅ", "uooaaUOOAA"))


EMPTY_VARIANTS_FOLDED = {_fold(v) for v in EMPTY_VARIANTS}


def empty_marker_kind(text: str) -> str | None:
    """Kas lehekülg on märgitud tühjaks?

    Tagastab:
      'exact'   – EMPTY_PAGE_MARKER (ümbritsev tühik lubatud), kõlblik
      'variant' – mõeldud tühjaks, aga vales vormis (paranda VUTT-is)
      'mixed'   – märgend koos muu tekstiga (kas leht pole tühi või jäi
                  märgend kogemata sisse)
      None      – tavaline leht
    """
    if text.strip() == EMPTY_PAGE_MARKER:
        return "exact"

    norm = re.sub(r"<[^>]+>", "", text)             # märgendid maha
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    if _fold(norm.strip("[](){}. ")) in EMPTY_VARIANTS_FOLDED:
        return "variant"
    if EMPTY_PAGE_MARKER.lower() in norm:
        return "mixed"
    return None


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


# Keelevalvur (VUTT ADR 0062 p 3): eesti, hispaania ja portugali keeles on tilde
# päris täht (õ, ñ, ã) — selliste teoste teksti ei teisendata makroniks.
VALVURIGA_KEELED = {"est", "et", "spa", "es", "por", "pt"}


def read_work_guarded(work_dir: Path) -> bool:
    """Kas teose `languages` sisaldab keelt, kus tilde on päris täht.
    Loetamatu metaandmestik → True: keelt teadmata ei tohi õ-d makroniks teha."""
    try:
        with open(work_dir / "_metadata.json", encoding="utf-8") as f:
            langs = json.load(f).get("languages") or []
    except Exception:
        return True
    return any(str(k).strip().lower() in VALVURIGA_KEELED for k in langs)


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
    makron_pages = 0
    makron_counts = Counter()
    makron_guarded = []    # valvuriga teose tildega lehed — käsitsi
    empty_pages = []        # korrektselt märgitud tühjad lehed
    empty_txt_pages = []    # Valmis, aga tekst puudub – kandidaat tühjaks
    sparse_pages = []       # vähese tekstiga lehed (nt ainult lk number)
    bad_empty = []          # tühjaks mõeldud, aga vales vormis
    mixed_empty = []        # märgend koos muu tekstiga

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
        guarded = read_work_guarded(work_dir)
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
                if keep:
                    # Tõenäoline tühi lehekülg, mis ootab VUTT-is märgendamist
                    empty_txt_pages.append(f"{work_dir.name}/{base}")
                continue

            # Tüübifilter alles siin, et loendur kajastaks päriselt kõlblikke
            # lehti, mitte ka neid, mis oleks niikuinii staatuse tõttu välja
            # kukkunud – muidu näitab statistika petlikult suuri numbreid.
            if not keep:
                type_pages[wtype] += 1
                continue

            cleaned = clean_markup(
                transcription, keep_marginalia_italics=KEEP_M_ITALICS)
            if cleaned != transcription:
                cleaned_markup += 1
            transcription = cleaned
            # Lühendusmärk → makron + prügi (VUTT ADR 0062; sama `puhasta` mis
            # 1. etapi CSV-del), välja arvatud valvuriga keeled
            if guarded:
                if "\u0303" in unicodedata.normalize("NFD", transcription):
                    makron_guarded.append(f"{work_dir.name}/{base}")
            else:
                puhas = puhasta(transcription, makron_counts)
                if puhas != transcription:
                    makron_pages += 1
                    transcription = puhas
            if not transcription:
                skipped_empty += 1
                continue

            # Tühja lehekülje märgendi kontroll
            kind = empty_marker_kind(transcription)
            page_id = f"{work_dir.name}/{base}"
            if kind == "exact":
                # Bait-bait sama string igal tühjal lehel – ainult nii õpib
                # mudel seda lõpetamismärgina.
                transcription = EMPTY_PAGE_MARKER
                empty_pages.append(page_id)
            elif kind == "variant":
                bad_empty.append((page_id, transcription.replace("\n", " ")[:60]))
                continue                      # vale vorm ei lähe andmestikku
            elif kind == "mixed":
                mixed_empty.append((page_id, transcription.replace("\n", " ")[:60]))
                continue
            elif ONLY_EMPTY:
                continue                      # tavaline leht, --only-empty jätab välja

            # Hõredad lehed: loenda alati, filtreeri ainult --max-chars puhul
            n_chars = plain_length(transcription)
            if n_chars <= SPARSE_LIMIT:
                sparse_pages.append((page_id, n_chars,
                                     transcription.replace("\n", " ")[:60]))
            elif MAX_CHARS is not None:
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
    print(f"  Lühendusmärk → makron + prügi: {makron_pages} lehte {dict(makron_counts)}"
          f" (valvuriga keel, tilde jäi: {len(makron_guarded)})")
    for lk in makron_guarded[:20]:
        print(f"    valvur: {lk}")
    excluded = {k: v for k, v in type_pages.items() if v}
    if excluded:
        print(f"  Vahele jäetud (vale tüüp): "
              + ", ".join(f"{k}={v}" for k, v in excluded.items()))
    # --- Tühjade lehekülgede aruanne -------------------------------------
    if ONLY_EMPTY:
        print(f"  --only-empty: ainult tühjaks märgitud lehed")
    print(f"  Tühjaks märgitud (õiges vormis '{EMPTY_PAGE_MARKER}'): "
          f"{len(empty_pages)}")
    for pid in empty_pages[:10]:
        print(f"        {pid}")
    if len(empty_pages) > 10:
        print(f"        ... ja veel {len(empty_pages) - 10}")

    # Hõredad lehed – vt --max-chars kommentaari failis ülal.
    if MAX_CHARS is not None:
        print(f"  --max-chars {MAX_CHARS}: ainult hõredad lehed")
    label = "valitud" if MAX_CHARS is not None else "andmestikus"
    print(f"  Hõredaid lehti ≤{SPARSE_LIMIT} märki ({label}): "
          f"{len(sparse_pages)}")
    for pid, n, txt in sorted(sparse_pages, key=lambda x: x[1])[:10]:
        print(f"        {n:4d} märki  {pid}: {txt!r}")
    if len(sparse_pages) > 10:
        print(f"        ... ja veel {len(sparse_pages) - 10}")
    if not sparse_pages and MAX_CHARS is None:
        print(f"        (mudel ei näe ühtegi näidet hõredast lehest – "
              f"vt SPIKKER.md)")

    if bad_empty:
        print(f"\n  !! {len(bad_empty)} lehte on tühjaks märgitud VALES VORMIS "
              f"– jäid välja.")
        print(f"     Paranda VUTT-is täpselt vormi: {EMPTY_PAGE_MARKER}")
        for pid, txt in bad_empty[:10]:
            print(f"        {pid}: {txt!r}")
        if len(bad_empty) > 10:
            print(f"        ... ja veel {len(bad_empty) - 10}")

    if mixed_empty:
        print(f"\n  !! {len(mixed_empty)} lehel on tühja lehe märgend KOOS muu "
              f"tekstiga – jäid välja.")
        print(f"     Kas leht pole tühi (eemalda märgend) või on märgend "
              f"kogemata sisse jäänud.")
        for pid, txt in mixed_empty[:10]:
            print(f"        {pid}: {txt!r}")
        if len(mixed_empty) > 10:
            print(f"        ... ja veel {len(mixed_empty) - 10}")

    # Loend on tüübifiltriga (ainult need, mis muidu andmestikku läheksid),
    # erinevalt üldloendurist skipped_empty.
    if empty_txt_pages:
        print(f"\n  NB! {len(empty_txt_pages)} Valmis lehel on transkriptsioon "
              f"tühi. Kui need on tühjad")
        print(f"      leheküljed, kirjuta VUTT-is sisuks {EMPTY_PAGE_MARKER} "
              f"– tühi string ei õpeta")
        print(f"      mudelile midagi ja leht jääb andmestikust välja.")
        for pid in empty_txt_pages[:10]:
            print(f"        {pid}")
        if len(empty_txt_pages) > 10:
            print(f"        ... ja veel {len(empty_txt_pages) - 10}")

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

    # --- Sihtfaili päis ja olemasolev sisu -------------------------------
    # Sihtandmestik võib olla kolmeveeruline (data/kurrent: failinimi,
    # transkriptsioon, allikas) või kaheveeruline (data/vutt). Loeme päise
    # olemasolevast failist, et mitte veergu kaotada.
    header = ["failinimi", "transkriptsioon"]
    if ALLIKAS_EXPLICIT:
        header.append("allikas")     # uus fail, aga allikaveerg on soovitud
    existing_files = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = ["failinimi", "transkriptsioon"]
            existing_files = {row[0] for row in reader if row}

        if "failinimi" not in header or "transkriptsioon" not in header:
            print(f"\nViga: {CSV_PATH} päis on ootamatu: {header}")
            sys.exit(1)

        if not APPEND and header != ["failinimi", "transkriptsioon"] and not FORCE:
            print(f"\nViga: {CSV_PATH} on olemas ja kolmeveeruline ({header}) – "
                  f"ülekirjutamine hävitaks {len(existing_files)} rida.")
            print("  Lisamiseks: --append   Ülekirjutamiseks: --force")
            sys.exit(1)

    if APPEND and "allikas" in header:
        print(f"\nLisan olemasolevale andmestikule: allikas={ALLIKAS}")

    # Loo väljundkataloog
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Kopeeri/skaleeri pildid ja kirjuta CSV
    counts = {"resized": 0, "copied": 0, "kept": 0}
    written = 0
    duplicates = 0
    mode = "a" if (APPEND and CSV_PATH.exists()) else "w"
    with open(CSV_PATH, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)

        for p in pairs:
            rel = f"images/{p['img_name']}"
            if APPEND and rel in existing_files:
                duplicates += 1
                continue

            dst = IMG_DIR / p["img_name"]
            if RESIZE_IMAGES:
                counts[prepare_image(p["img_src"], dst)] += 1
            elif not dst.exists():
                shutil.copy2(p["img_src"], dst)
                counts["copied"] += 1
            else:
                counts["kept"] += 1

            values = {
                "failinimi": rel,
                "transkriptsioon": p["transkriptsioon"],
                "allikas": ALLIKAS,
            }
            writer.writerow([values.get(col, "") for col in header])
            written += 1

    if duplicates:
        print(f"  Juba andmestikus, vahele jäetud: {duplicates}")

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
    print(f"  CSV: {CSV_PATH} ({written} rida "
          + ("juurde lisatud)" if APPEND else "kirjutatud)"))

    # Päritolu: uus mudel tehakse ~korra kuus, seega neli nädalat hiljem ei mäleta
    # keegi, millise andmeseisu pealt see treeniti. Kolm rida, mis selle vastavad.
    # --append puhul lisame kirje, mitte ei kirjuta üle: sihtandmestik on
    # ehitatud mitmest jooksust ja iga jooks peab jälje jätma.
    source_path = OUT_DIR / "SOURCE.txt"
    with open(source_path, "a" if APPEND else "w", encoding="utf-8") as f:
        if APPEND:
            f.write("\n")
        f.write(f"raw_dir: {RAW_DIR}\n")
        f.write(f"ehitatud: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"lehti: {written}  (--type {MATERIAL}"
                + (", + tundmatud" if INCLUDE_UNKNOWN else "")
                + (", --only-empty" if ONLY_EMPTY else "")
                + (f", --append allikas={ALLIKAS}" if APPEND else "")
                + (", --resize" if RESIZE_IMAGES else "")
                + (", --keep-m-italics" if KEEP_M_ITALICS else "") + ")\n")
    print(f"  Päritolu: {source_path}")

    if OUT_EXPLICIT:
        print(f"\nJärgmine samm: kontrolli {CSV_PATH} ja treeni sihtandmestiku "
              f"skriptiga.")
    else:
        print(f"\nJärgmine samm: python scripts/train_markup.py [--test]")


if __name__ == "__main__":
    main()
