#!/usr/bin/env python3
"""Dresdner Hofdiarium 1665 uuesti (2026-10-04), DTA konverteri reeglitega.

Vana build_dresdner_tei_dataset.py lamendas TEI: tabeli veerud segunesid
ühele reale, pb piiril lekkis järgmise lehe tekst sisse (pb on sageli
persName-i sees) ja marginaalid (note) visati ära.
Siin: build_dta_kosmos.Lehed (pb mis tahes sügavusel), lisaks
  <ex> (toimetaja laiend) → „." — lehel on laiendi kohal lühendusmärk
     (Churf. durchl.; nii kirjutasid transkribeerijad ka tabelilahtrites),
     kui järgmine märk pole juba . või :; <figure>/<desc> välja
  tabeliga leht välja (veerud põimuvad, reajärjekord pole üheselt määratud)
  ſ JÄÄB (nagu teistes XVII saj allikates: dresdner, senats)
  ainult pb type="diaryEntry" (nagu vana ehitaja)
Pildid: olemasolevad data/kurrent_v4 dresdner1665 failid hardlink-itakse
(tif-numbri järgi), puuduvad SLUB-ist.
Väljund: data/dresdner_v2/{images/, metadata.csv, projektid.csv, SOURCE.txt}
Käivitus: venv/bin/python scripts/build_dresdner_v2.py [--dry-run | --naita <tif-nr>]
"""
import csv, io, os, re, sys, time, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import build_dta_kosmos as dta                                        # noqa: E402

DRY = "--dry-run" in sys.argv
TEI = Path("data/raw_xml/hofdiarium1665.xml")   # Zenodo 14932508 Release 2
OLD = Path("data/kurrent_v4")
OUT = Path("data/dresdner_v2")
ALLIKAS = "dresdner_1665"
MIN_LINES = 3                                   # = vana ehitaja
NS = dta.NS


class Lehed(dta.Lehed):
    def __init__(self):
        super().__init__()
        self.url, self.tyyp = {}, {}

    def walk(self, el):
        t = dta.tag(el)
        if t == "pb":
            m = re.search(r"(\d+)\.tif", el.get("facs", ""))
            self.cur = int(m.group(1)) if m else None
            if self.cur is not None:
                self.lehed.setdefault(self.cur, [])
                self.url[self.cur] = el.get("facs")
                self.tyyp[self.cur] = el.get("type", "")
        elif t == "ex":
            self.emit("\x01")                         # lühendusmärgi koht, vt lehe_tekst
        elif t in ("figure", "desc"):
            pass
        else:
            return super().walk(el)
        if el.tail:
            self.emit(el.tail)


def lehe_tekst(tykid):
    s = "".join("\x00" if x == "\n" else x.replace("\n", " ") for x in tykid)
    s = re.sub(r"\x01+(?=\s*[.:])", "", s)
    s = re.sub(r"\s*\x01+", ".", s)                   # <ex> → „." vahetult sõna järel
    read = [re.sub(r"\s+", " ", r).strip() for r in s.split("\x00")]
    return "\n".join(r for r in read if r)


def loe():
    root = ET.parse(TEI).getroot()
    lh = Lehed()
    lh.walk(root.find(f"{NS}text"))
    tul = {}
    for n, tykid in lh.lehed.items():
        if lh.tyyp.get(n) != "diaryEntry":
            continue
        t = lehe_tekst(tykid)
        p = lh.halvad.get(n)
        if not p and len(t.split("\n")) < MIN_LINES:
            p = "lühike"
        tul[n] = (t, p, lh.url[n])
    return tul


def main():
    tul = loe()
    if "--naita" in sys.argv:
        t, p, u = tul[int(sys.argv[sys.argv.index("--naita") + 1])]
        print(f"[{p or chr(79)+chr(75)}] {u}\n{t}")
        return
    st = Counter(p or "OK" for _, p, _ in tul.values())
    print(f"diaryEntry lehti {len(tul)}: {dict(st)}")
    if DRY:
        return
    if OUT.exists():
        sys.exit(f"{OUT} on juba olemas")
    olemas = {int(m.group(1)): OLD / r["failinimi"]
              for r in csv.DictReader(open(OLD / "metadata.csv", encoding="utf-8"))
              if r["allikas"] == ALLIKAS and (m := re.search(r"_(\d+)\.jpg$", r["failinimi"]))}
    (OUT / "images").mkdir(parents=True)
    rows = []
    for n, (t, p, url) in sorted(tul.items()):
        if p:
            continue
        name = f"images/dresdner1665_{n:08d}.jpg"
        if n in olemas:
            os.link(olemas[n], OUT / name)
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "VUTT-HTR/1.0 (Tartu Ülikool)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    Image.open(io.BytesIO(r.read())).convert("RGB").save(OUT / name, "JPEG", quality=90)
                time.sleep(0.3)
            except Exception as e:
                print(f"  pildi viga {n}: {e}", flush=True)
                continue
        rows.append((name, t, ALLIKAS))
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "transkriptsioon", "allikas"]); w.writerows(rows)
    with open(OUT / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "projekt"]); w.writerows((r[0], "hofdiarium1665") for r in rows)
    (OUT / "SOURCE.txt").write_text(
        f"ehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_dresdner_v2.py\n"
        f"Zenodo 14932508 Release 2 (SLUB Mscr K80, Hofdiarium 1665), CC BY 4.0\n"
        f"lehti: {len(rows)}  {dict(st)}  (pilte olemas {sum(1 for r in rows if int(r[0][-12:-4]) in olemas)})\n",
        encoding="utf-8")
    print(f"valmis: {OUT} ({len(rows)} lk)")


if __name__ == "__main__":
    main()
