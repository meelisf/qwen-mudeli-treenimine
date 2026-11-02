#!/usr/bin/env python3
"""Tühjade TextLine'ide audit PAGE XML-is (SEIS §6.11).

`parse_pagexml` jätab teksti(ta) rea VAIKSELT vahele — pilt näitab rida, GT-s
seda pole. Escheri XML-id on CITlabi „Matcher" automaatjoondus (editsioonitekst
ridadele): kus joondus ebaõnnestus, jäi rida tühjaks. See skript mõõdab iga
lehe tühjade ridade osakaalu ja looja (Creator), et otsustada, mis jääb
treeningmaterjali.

Kaks režiimi:
  (vaikimisi) kohalik HF cache: xix + hanse → data/kurrent_xix_audit/tuhjad_read.csv
  --kaug      Bullinger, AAEB, Königsfelden otse Hugging Face'ist, AINULT
              xml_content veerg (parquet range-päringud, pilte ei laadita)
              → data/kurrent_xix_audit/tuhjad_read_kaug.csv
Väljal `treeningus` on andmestiku failinimi (data/kurrent), kui leht on seal.

Käivitus: venv/bin/python scripts/tuhjad_read_audit.py [--kaug]
"""
import csv, glob, re, sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

KAUG = "--kaug" in sys.argv
HF = Path.home() / "_kustutamiseks_20261002/hf_cache"
ALLIKAD = {
    "xix": "datasets--dh-unibe--image-text_kurrent-xix",
    "hanse_xvi": "datasets--fgho--hanse-kurrent-xvi-rawxml",
    "hanse_xvii": "datasets--fgho--hanse-kurrent-xvii-rawxml",
}
# allikas → (HF repo, failinime lühend build_kurrent_dataset.py _DS_SHORT järgi)
ALLIKAD_KAUG = {
    "bullinger_autoren": ("dh-unibe/image-text_bullinger-autoren", "bullinger_au"),
    "aaeb_xiv_xvii": ("dh-unibe/image-text_aaeb-xiv-xvii", "aaeb_xiv_xvi"),
    "koenigsfelden_adhr": ("dh-unibe/image-text_koenigsfelden-adhr-colmar", "koenigsfelde"),
}
OUT = Path("data/kurrent_xix_audit") / ("tuhjad_read_kaug.csv" if KAUG else "tuhjad_read.csv")
LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)
CREATOR = re.compile(r"<Creator>(.*?)</Creator>", re.S)

csv.field_size_limit(sys.maxsize)


def rea_tekst(tl):
    u = UNI.findall(WORD.sub("", tl))        # rea enda TextEquiv, mitte sõnade oma
    return u[-1].strip() if u else ""


def looja(xml):
    m = CREATOR.search(xml)
    c = m.group(1) if m else ""
    if "Matcher" in c:
        return "matcher"
    return "transkribus" if "Transkribus" in c else (c.split(":")[0][:40] or "?")


def stem_of(filename):
    return re.sub(r"[^\w\-]", "_", Path(filename).stem)


def treeningus_xix():
    """(projekt, tüvi) → andmestiku failinimi; xix projektid.csv-st."""
    t = {}
    for r in csv.DictReader(open("data/kurrent/projektid.csv", encoding="utf-8")):
        m = re.match(r"images/\d+_xixr_(.+)\.jpg$", r["failinimi"])
        if m:
            t[(r["projekt"], m.group(1))] = r["failinimi"]
    return t


def treeningus_kaug():
    """(lühend, tüvi) → andmestiku failinimi; metadata.csv-st."""
    t = {}
    lyhid = [v[1] for v in ALLIKAD_KAUG.values()]
    for r in csv.reader(open("data/kurrent/metadata.csv", encoding="utf-8")):
        m = re.match(r"images/\d+_(.+)\.jpg$", r[0])
        if m:
            for ly in lyhid:
                if m.group(1).startswith(ly + "_"):
                    t[(ly, m.group(1)[len(ly) + 1:])] = r[0]
    return t


def failid():
    """(allikas, lühend, parquet-fail avatav pq-le) generaator."""
    if KAUG:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        for allikas, (repo, ly) in ALLIKAD_KAUG.items():
            fl = sorted(fs.glob(f"datasets/{repo}/data/**/*.parquet"))
            print(f"{allikas}: {len(fl)} faili", flush=True)
            for i, f in enumerate(fl):
                if i % 50 == 0:
                    print(f"  {i}/{len(fl)}", flush=True)
                yield allikas, ly, fs.open(f, block_size=1 << 20)
            print(f"{allikas} tehtud", flush=True)
    else:
        for allikas, d in ALLIKAD.items():
            for pf in sorted(glob.glob(str(HF / d / "snapshots/*/data/**/*.parquet"), recursive=True)):
                yield allikas, None, pf
            print(f"{allikas} tehtud", flush=True)


def main():
    tr = treeningus_kaug() if KAUG else treeningus_xix()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proj = defaultdict(lambda: [0, 0, 0, 0, 0, set()])   # lk, ridu, tühje, lk>0 tühja, treeningus, loojad
    with open(OUT, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["allikas", "projekt", "failinimi", "ridu", "tuhje", "osakaal", "looja", "treeningus"])
        for allikas, ly, src in failid():
            pf = pq.ParquetFile(src)
            for b in pf.iter_batches(batch_size=64, columns=["xml_content", "filename", "project_name"]):
                for r in b.to_pylist():
                    x = r["xml_content"] or ""
                    read = [rea_tekst(tl) for tl in LINE.findall(x)]
                    n, e = len(read), sum(1 for s in read if not s)
                    p = r["project_name"] or "?"
                    if KAUG:
                        trn = tr.get((ly, stem_of(r["filename"])), "")
                    else:
                        trn = tr.get((p, stem_of(r["filename"])[:60]), "") if allikas == "xix" else ""
                    lj = looja(x)
                    w.writerow([allikas, p, r["filename"], n, e, f"{e / n:.3f}" if n else "", lj, trn])
                    s = proj[(allikas, p)]
                    s[0] += 1; s[1] += n; s[2] += e; s[3] += e > 0; s[4] += bool(trn); s[5].add(lj)
    print("\n| allikas | projekt | lk | treeningus | tühje ridu | lk-l tühi rida | looja |")
    print("|---|---|---|---|---|---|---|")
    for (a, p), s in sorted(proj.items(), key=lambda kv: -kv[1][2] / max(kv[1][1], 1)):
        if s[2]:
            print(f"| {a} | {p} | {s[0]} | {s[4]} | {s[2] / max(s[1], 1):.1%} | {s[3]} ({s[3] / s[0]:.0%}) | {','.join(sorted(s[5]))} |")
    print(f"\nTühja reata projekte: {sum(1 for s in proj.values() if not s[2])} / {len(proj)}")


if __name__ == "__main__":
    main()
