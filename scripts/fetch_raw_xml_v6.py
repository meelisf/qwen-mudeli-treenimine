#!/usr/bin/env python3
"""Kurrent v6: hanse XVI/XVII ja kurrent-xix PAGE XML uuesti alla (ainult xml_content).

hf_cache kustutati pärast v2/v4 ehitust; veerise <m>-iks tegemiseks on vaja
regioonide geomeetriat. Pilte EI laadita (on data/kurrent_v5m-is) — HfFileSystem +
pyarrow loevad parquet'ist ainult xml_content/filename veergu (range-päringud).

  venv/bin/python scripts/fetch_raw_xml_v6.py [hanse16|hanse17|xix ...]
→ data/raw_xml/{hanse_xvi,hanse_xvii,kurrent_xix}/<projekt>/<failinimi>.xml
"""
import csv, re, sys, time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

RAW = Path("data/raw_xml")
PROJ = Path("data/kurrent_v5m/projektid.csv")
fs = HfFileSystem()
PROJ_FILTER = None


def tyvi(nimi):
    return re.sub(r"[^\w\-]", "_", Path(nimi or "").stem)


def kirjuta(siht, projekt, nimi, xml):
    p = RAW / siht / tyvi(projekt or "_") / f"{tyvi(nimi)}.xml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(xml, encoding="utf-8")


def loe(failid, veerud, kas_vaja, siht, projekt_veerust=None, projekt=None):
    n = 0
    for f in failid:
        for katse in range(5):
            try:
                pf = pq.ParquetFile(fs.open(f, block_size=1 << 20))
                for b in pf.iter_batches(batch_size=64, columns=veerud):
                    for r in b.to_pylist():
                        if r.get("xml_content") and kas_vaja(r):
                            kirjuta(siht, r.get(projekt_veerust) if projekt_veerust else projekt,
                                    r["filename"], r["xml_content"]); n += 1
                break
            except Exception as e:                 # võrk kõigub: proovi uuesti
                print(f"  viga {f}: {e} — katse {katse + 1}/5", flush=True); time.sleep(10 * (katse + 1))
        print(f"  {f.split('/')[-2]}/{f.split('/')[-1]}: kokku {n}", flush=True)
    return n


def hanse(repo, siht):
    failid = sorted(fs.glob(f"datasets/{repo}/**/*.parquet"))
    print(f"{repo}: {len(failid)} parquet'it", flush=True)
    veerud = [c for c in ("xml_content", "filename", "project_name")
              if c in pq.ParquetFile(fs.open(failid[0])).schema_arrow.names]
    return loe(failid, veerud, lambda r: True, siht, projekt_veerust="project_name" if "project_name" in veerud else None)


def xix():
    # ainult v5m-is olevad projektid/failid: pildinimi = <nr>_xixr_<failinime tüvi[:60]>.jpg
    vaja = defaultdict(set)
    for r in csv.reader(open(PROJ, encoding="utf-8")):
        if "_xixr_" in r[0] and len(r) > 1 and r[1]:
            vaja[r[1]].add(re.sub(r"^images/\d+_xixr_", "", r[0])[:-4])
        elif r[0].startswith("images/esch_") and len(r) > 1 and r[1]:    # build_escher_lisa.py nimekuju
            vaja[r[1]].add(r[0][len("images/esch_"):-4])
    print(f"kurrent-xix: {len(vaja)} projekti, {sum(map(len, vaja.values()))} lehte", flush=True)
    n = 0
    for proj in sorted(vaja):
        if PROJ_FILTER and proj != PROJ_FILTER:
            continue
        failid = sorted(fs.glob(f"datasets/dh-unibe/image-text_kurrent-xix/data/train/{proj}/*.parquet"))
        n += loe(failid, ["xml_content", "filename"], lambda r, p=proj: tyvi(r["filename"])[:60] in vaja[p],
                 "kurrent_xix", projekt=proj)
    return n


if __name__ == "__main__":
    mida = sys.argv[1:] or ["hanse16", "hanse17", "xix"]
    if "escher" in mida:                       # ainult Escher (lisati 14.10 pärast põhijooksu)
        PROJ_FILTER = "TRAIN_CITlab_Escher_M1"
    if "hanse16" in mida:
        print("hanse XVI:", hanse("fgho/hanse-kurrent-xvi-rawxml", "hanse_xvi"), flush=True)
    if "hanse17" in mida:
        print("hanse XVII:", hanse("fgho/hanse-kurrent-xvii-rawxml", "hanse_xvii"), flush=True)
    if "xix" in mida or "escher" in mida:
        print("kurrent-xix:", xix(), flush=True)
