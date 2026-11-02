#!/usr/bin/env python3
"""Kurrendi andmestik v2 (2026-10-02): mitmekesisus Zürichi puhtandi asemel.

Ehitab UUE kausta `data/kurrent_uus/` (images/, metadata.csv, holdout.txt,
projektid.csv, SOURCE.txt). Vana `data/kurrent/` jääb PUUTUMATA: säilitatavad
pildid hardlink'itakse, mitte ei liigutata. Vahetus (vana → prügikasti, uus →
`data/kurrent`) on eraldi käsitsi samm pärast kontrolli.

Koostis:
  - vanast: kõik allikad, AGA kurrent_xix (= ainult Zürich MM_1_001…033)
    piiratakse ZURICH_CAP-ini; holdout-lehed jäävad alati alles
  - kurrent_xix mitte-Zürichi projektid `data/kurrent_xix_audit/unique_pages.csv`
    järgi (dedup'itud, vt docs/kurrent-andmestikud.md „Ülevaatus 2026-10-02"):
    Escher piiratud, 1900+ projekti kohta piiratud
  - fgho/hanse-kurrent-xvii-rawxml tervikuna

Tekst tuleb SAMADEST `parse_pagexml`-idest mis vanadel allikatel (rea tasandi
TextEquiv, teisendusteta), MIN_LINES samad mis vanadel skriptidel.

Käivitus:
  venv/bin/python scripts/build_kurrent_v2.py --dry-run   # ainult loendab
  venv/bin/python scripts/build_kurrent_v2.py
"""
import csv, glob, io, os, random, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_kurrent_dataset import parse_pagexml as parse_xix          # noqa: E402
from build_hanse_dataset import parse_pagexml as parse_hanse          # noqa: E402

DRY = "--dry-run" in sys.argv
SEED = 3407
ZURICH_CAP = 1000
ESCHER_CAP = 1000
CAP_1900 = 50            # projekti kohta; VUTT-il XX saj materjali peaaegu pole
HOLDOUT_PER_NEW = 10
HOLDOUT_MIN_CHARS = 200
MIN_LINES_XIX = 5        # = build_kurrent_dataset.MIN_LINES
MIN_LINES_HANSE = 3      # = build_hanse_dataset.MIN_LINES

OLD = Path("data/kurrent")
NEW = Path("data/kurrent_uus")
HF = Path.home() / ".cache/huggingface/hub"
XIX_TRAIN = glob.glob(str(HF / "datasets--dh-unibe--image-text_kurrent-xix/snapshots/*/data/train"))[0]
HANSE17 = glob.glob(str(HF / "datasets--fgho--hanse-kurrent-xvii-rawxml/snapshots/*/data/train"))[0]
UNIQUE = Path("data/kurrent_xix_audit/unique_pages.csv")

# Perioodist allikanimi (3. veerg); täpne projekt läheb projektid.csv-sse
PERIOD_ALLIKAS = {"1750-99": "xix_read_1750_99", "1800-49": "xix_read_1800_49",
                  "1850-99": "xix_read_1850_99", "1900+": "xix_read_1900",
                  "?": "xix_read_dateerimata"}

# Vanad read, millel 3. veerg puudub (SEIS.md „kosmeetiline andmeviga")
ALLIKAS_FROM_PREFIX = {"aaeb": "aaeb_xiv_xvii", "jonkopings": "jonkopings_seg"}

rng = random.Random(SEED)


def old_allikas(row):
    if len(row) >= 3 and row[2]:
        return row[2]
    pref = re.sub(r"^images/\d+_", "", row[0]).split("_")[0]
    return ALLIKAS_FROM_PREFIX[pref]          # KeyError = tundmatu, peatu


def save_jpg(raw_bytes, path):
    Image.open(io.BytesIO(raw_bytes)).convert("RGB").save(path, "JPEG", quality=90)


def main():
    if NEW.exists() and not DRY:
        sys.exit(f"{NEW} on juba olemas — eemalda või nimeta ümber enne uut jooksu")
    if not DRY:
        (NEW / "images").mkdir(parents=True)

    out_rows = []            # (failinimi, tekst, allikas)
    projekt = {}             # failinimi -> projekt (uued read)

    # --- 1. Vana andmestik ------------------------------------------------
    old_holdout_lines = [l.rstrip("\n") for l in open(OLD / "holdout.txt", encoding="utf-8")]
    holdout_names = {os.path.basename(l.split("\t")[-1]) for l in old_holdout_lines
                     if l and not l.startswith("#")}
    rows = list(csv.reader(open(OLD / "metadata.csv", encoding="utf-8")))[1:]
    zurich = [r for r in rows if old_allikas(r) == "kurrent_xix"]
    z_hold = [r for r in zurich if os.path.basename(r[0]) in holdout_names]
    z_rest = [r for r in zurich if os.path.basename(r[0]) not in holdout_names]
    z_keep = {r[0] for r in z_hold + rng.sample(z_rest, ZURICH_CAP - len(z_hold))}
    for r in rows:
        a = old_allikas(r)
        if a == "kurrent_xix":
            if r[0] not in z_keep:
                continue
            a = "kurrent_xix_zurich"           # nimi ütleb nüüd, mis see on
        out_rows.append((r[0], r[1], a))
        if not DRY:
            os.link(OLD / r[0], NEW / r[0])
    print(f"vana: {len(rows)} → {len(out_rows)} (Zürich {len(zurich)} → {len(z_keep)})", flush=True)
    # vutt_horedad read on loendurita (VUTT-i failinimed) — need jäetakse välja
    counter = max(int(m.group(1)) for r in rows if (m := re.match(r"images/(\d+)_", r[0]))) + 1

    # --- 2. kurrent_xix mitte-Zürich ---------------------------------------
    sel = defaultdict(list)
    period_of = {}
    for r in csv.DictReader(open(UNIQUE, encoding="utf-8")):
        sel[r["project"]].append(r["filename"])
        period_of[r["project"]] = r["period"]
    want = {}
    for proj, files in sel.items():
        files = sorted(files)
        cap = ESCHER_CAP if proj == "TRAIN_CITlab_Escher_M1" else \
              CAP_1900 if period_of[proj] == "1900+" else None
        if cap and len(files) > cap:
            files = rng.sample(files, cap)
        for f in files:
            want[(proj, f)] = PERIOD_ALLIKAS[period_of[proj]]
    stats = Counter()
    for proj in sorted(sel):
        for pf in sorted(glob.glob(f"{XIX_TRAIN}/{proj}/*.parquet")):
            for batch in pq.ParquetFile(pf).iter_batches(
                    batch_size=16, columns=["xml_content", "filename"] + ([] if DRY else ["image"])):
                for r in batch.to_pylist():
                    key = (proj, r["filename"])
                    if key not in want:
                        continue
                    want.pop(key)             # sama leht võib shard'ides korduda
                    text = parse_xix(r["xml_content"] or "")
                    if not text or sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES_XIX:
                        stats["xix_liiga_lühike"] += 1
                        continue
                    stem = re.sub(r"[^\w\-]", "_", Path(r["filename"]).stem)[:60]
                    name = f"images/{counter:05d}_xixr_{stem}.jpg"
                    counter += 1
                    if not DRY:
                        try:
                            save_jpg(r["image"]["bytes"], NEW / name)
                        except Exception:
                            stats["xix_pildi_viga"] += 1
                            continue
                    allikas = PERIOD_ALLIKAS[period_of[proj]]
                    out_rows.append((name, text, allikas))
                    projekt[name] = proj
                    stats[allikas] += 1
        print(f"  {proj}: kokku {sum(v for k, v in stats.items() if k.startswith('xix_read'))}", flush=True)
    stats["xix_leidmata"] = len(want)

    # --- 3. hanse-kurrent-xvii --------------------------------------------
    for pf in sorted(glob.glob(f"{HANSE17}/*/*.parquet")):
        for batch in pq.ParquetFile(pf).iter_batches(
                batch_size=8, columns=["xml_content", "filename", "project_name"] + ([] if DRY else ["image"])):
            for r in batch.to_pylist():
                text = parse_hanse(r["xml_content"] or "")
                if not text or sum(1 for l in text.split("\n") if l.strip()) < MIN_LINES_HANSE:
                    stats["hanse17_liiga_lühike"] += 1
                    continue
                stem = re.sub(r"[^\w\-]", "_", r["filename"] or "")[:60]
                name = f"images/{counter:05d}_hanse17_{stem}.jpg"
                counter += 1
                if not DRY:
                    try:
                        save_jpg(r["image"]["bytes"], NEW / name)
                    except Exception:
                        stats["hanse17_pildi_viga"] += 1
                        continue
                out_rows.append((name, text, "hanse_kurrent_xvii"))
                projekt[name] = r["project_name"]
                stats["hanse_kurrent_xvii"] += 1
    print("statistika:", dict(stats), flush=True)

    # --- 4. Holdout: vana 73 muutmata + uued allikad -----------------------
    new_allikad = sorted({a for _, _, a in out_rows if a.startswith("xix_read") or a == "hanse_kurrent_xvii"})
    hold_new = []
    for a in new_allikad:
        cand = [r for r in out_rows if r[2] == a and len(r[1]) >= HOLDOUT_MIN_CHARS]
        # eri projektidest, kui võimalik
        by_proj = defaultdict(list)
        for r in cand:
            by_proj[projekt[r[0]]].append(r)
        picks = []
        projs = sorted(by_proj)
        rng.shuffle(projs)
        while len(picks) < HOLDOUT_PER_NEW and any(by_proj.values()):
            for p in projs:
                if by_proj[p] and len(picks) < HOLDOUT_PER_NEW:
                    picks.append(by_proj[p].pop(rng.randrange(len(by_proj[p]))))
        hold_new += [(a, r[0]) for r in picks]

    # Vanade sildita holdout-ridade silt parandatakse, failinimi jääb
    fixed_old = []
    allikas_by_name = {os.path.basename(r[0]): r[2] for r in out_rows}
    for l in old_holdout_lines:
        if l.startswith("(sildita)\t"):
            f = l.split("\t", 1)[1]
            l = f"{allikas_by_name[os.path.basename(f)]}\t{f}"
        elif l.startswith("kurrent_xix\t"):
            l = "kurrent_xix_zurich\t" + l.split("\t", 1)[1]
        fixed_old.append(l)

    print(f"KOKKU {len(out_rows)} rida; holdout {len(holdout_names)} vana + {len(hold_new)} uut")
    print("allikad:", Counter(a for _, _, a in out_rows).most_common())
    if DRY:
        return

    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["failinimi", "transkriptsioon", "allikas"])
        w.writerows(out_rows)
    with open(NEW / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["failinimi", "projekt"])
        w.writerows(sorted(projekt.items()))
    with open(NEW / "holdout.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_old) + "\n")
        f.write(f"# --- Lisatud {datetime.now():%Y-%m-%d} build_kurrent_v2.py "
                f"(seed={SEED}, {HOLDOUT_PER_NEW}/uus allikas, min_chars={HOLDOUT_MIN_CHARS}) ---\n")
        f.writelines(f"{a}\t{n}\n" for a, n in hold_new)
    src = (OLD / "SOURCE.txt").read_text(encoding="utf-8") if (OLD / "SOURCE.txt").exists() else ""
    (NEW / "SOURCE.txt").write_text(src + (
        f"\nehitatud: {datetime.now().isoformat(timespec='seconds')}  scripts/build_kurrent_v2.py\n"
        f"read: {len(out_rows)}  Zürich {len(zurich)}→{len(z_keep)}  Escher≤{ESCHER_CAP}  "
        f"1900+≤{CAP_1900}/projekt  + hanse_kurrent_xvii\n"), encoding="utf-8")
    print("VALMIS")


if __name__ == "__main__":
    main()
