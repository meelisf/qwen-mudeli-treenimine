#!/usr/bin/env python3
"""Humboldti Kosmos-Nachschriften (DTA, 1827–29) lehekaupa treeningandmeteks.

Miks: xix-i parthey/hufeland/nn_msgermqu on CITlabi „Matcher" automaatjoondus —
ebaõnnestunud read jäid tühjaks ja GT on osaline (vt tuhjad_read_audit.py).
DTA TEI kannab sama käsikirja TÄIELIKKU lehe teksti; treenime lehe tasemel,
nii et reajoondust ei ole vaja. Lisaks 5 käsikirja, mida xix-is pole.
Litsents CC BY 4.0. Loend: HU Berlin „Nachschriften der Kosmos-Vorträge".

TEI → diplomaatiline tekst:
  choice → abbr | orig | sic (mitte expan/reg/corr); üksik expan/reg/corr/supplied välja
  del #s jääb (läbikriipsutus on pildil loetav); del #ow/#erased välja (ülekirjutatud)
  add, unclear, hi, fw, marginaal-note jäävad; note type=editorial välja; metamark välja
  ſ → s (ülejäänud Kurrendi GT-s ſ-i ei ole); rea lõpu „-" jääb nagu DTA-s
Leht jääb VÄLJA, kui seal on gap (loetamatu/kadunud), tabel, U+FFFC (esitamatu
lühendusmärk) või vähem kui MIN_LINES rida.

Pildid: media.dwds.de 1600px (facs-numbri järgi), vahemällu data/dta_tei/img/.
Väljund: data/dta_kosmos/{images/, metadata.csv, projektid.csv, SOURCE.txt}

Käivitus:
  venv/bin/python scripts/build_dta_kosmos.py --dry-run   # ainult loendab, pilte ei laadi
  venv/bin/python scripts/build_dta_kosmos.py --grupp lisa  # Geusau + Sanders → data/dta_lisa
  venv/bin/python scripts/build_dta_kosmos.py
  venv/bin/python scripts/build_dta_kosmos.py --naita parthey_msgermqu1711_1828 673
"""
import csv, os, re, sys, time, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

DRY = "--dry-run" in sys.argv
TEI_DIR = Path("data/dta_tei")
IMG_CACHE = TEI_DIR / "img"
MIN_LINES = 5
IMG_URL = "https://media.dwds.de/dta/images/{id}/{id}_{n:04d}_1600px.jpg"
KOSMOS = [
    "parthey_msgermqu1711_1828", "hufeland_privatbesitz_1829",
    "nn_msgermqu2124_1827", "nn_msgermqu2345_1827",       # xix-is Matcheriga
    "libelt_hs6623ii_1828", "patzig_msgermfol841842_1828",
    "willisen_humboldt_1827", "nn_oktavgfeo79_1828", "nn_n0171w1_1828",
]
# --grupp lisa (04.10): DTA täiskorpuse käsikirjad (TEI: data/raw_xml/dta_komplett,
# lingitud data/dta_tei/ alla). Geusau = Heinrich XI. Reuß reisipäevik 1740 (kiire
# Kurrent, prantsuse kohanimed antiikvas); Sanders = Daniel Sandersi kirjad 1859–80.
GRUPID = {
    "kosmos": (Path("data/dta_kosmos"), "Kosmos-Nachschriften 1827–29",
               {t: "dta_kosmos_1827" for t in KOSMOS}),
    "lisa": (Path("data/dta_lisa"), "käsikirjad: Geusau 1740, Sanders 1859–80",
             {"geusau_reisetagebuchHeinrichxiReuss_1740": "dta_geusau_1740",
              **{p.stem: "dta_sanders_1860" for p in sorted(TEI_DIR.glob("sanders_*.xml"))}}),
}
GRUPP = sys.argv[sys.argv.index("--grupp") + 1] if "--grupp" in sys.argv else "kosmos"
OUT, KIRJELDUS, TEOSED = GRUPID[GRUPP]
NS = "{http://www.tei-c.org/ns/1.0}"
EI_SISU = {"expan", "reg", "corr", "supplied", "metamark", "figDesc", "graphic", "teiHeader"}


def tag(el):
    return el.tag.replace(NS, "") if isinstance(el.tag, str) else ""


class Lehed:
    """Kogub teksti `pb`-de vahel; pb võib olla sügaval p/div sees."""

    def __init__(self):
        self.lehed = {}          # facs-nr → [tükid]
        self.halvad = {}         # facs-nr → põhjus
        self.cur = None

    def emit(self, s):
        if self.cur is not None:
            self.lehed[self.cur].append(s)

    def halb(self, pohjus):
        if self.cur is not None:
            self.halvad.setdefault(self.cur, pohjus)

    def walk(self, el):
        t = tag(el)
        if t == "pb":
            m = re.search(r"(\d+)", el.get("facs", ""))
            self.cur = int(m.group(1)) if m else None
            if self.cur is not None:
                self.lehed.setdefault(self.cur, [])
        elif t == "lb":
            self.emit("\n")
        elif t == "space":
            self.emit(" ")
        elif t == "gap":
            self.halb("gap")
        elif t in ("table", "row", "cell"):
            self.halb("tabel")
            self._sisu(el)
        elif t in EI_SISU:
            pass
        elif t == "choice":
            for c in el:
                if tag(c) in ("abbr", "orig", "sic"):
                    self.walk(c)
                    break
        elif t == "del" and re.search(r"#(ow|erased)", el.get("rendition", "")):
            pass
        elif t == "note" and (el.get("type") == "editorial" or not el.get("place")):
            pass
        elif t in ("note", "fw"):
            self.emit("\n")
            self._sisu(el)
            self.emit("\n")
        else:
            self._sisu(el)
        if el.tail:                                  # tail kuulub vanemale, aga järjekorras siia
            self.emit(el.tail)

    def _sisu(self, el):
        if el.text:
            self.emit(el.text)
        for c in el:
            self.walk(c)


def puhasta(tykid):
    s = "".join(tykid)
    s = re.sub(r"[ \t\r\n]+", lambda m: "\n" if "\x00" in m.group(0) else " ", s.replace("\n", " "))
    return s


def lehe_tekst(tykid):
    # XML-i vormindusreavahetus = tühik; ainult <lb/> (emit "\n" → \x00) teeb rea
    s = "".join("\x00" if x == "\n" else x.replace("\n", " ") for x in tykid)
    s = s.replace("ſ", "s")
    read = [re.sub(r"\s+", " ", r).strip() for r in s.split("\x00")]
    return "\n".join(r for r in read if r)


def loe_teos(teos_id):
    root = ET.parse(TEI_DIR / f"{teos_id}.xml").getroot()
    text = root.find(f"{NS}text")
    lh = Lehed()
    lh.walk(text)
    tulemus = {}
    for n, tykid in lh.lehed.items():
        t = lehe_tekst(tykid)
        pohjus = lh.halvad.get(n)
        if not pohjus and "￼" in t:
            pohjus = "U+FFFC"
        if not pohjus and len(t.split("\n")) < MIN_LINES:
            pohjus = "lühike"
        tulemus[n] = (t, pohjus)
    return tulemus


def lae_pilt(teos_id, n):
    p = IMG_CACHE / teos_id / f"{n:04d}.jpg"
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(IMG_URL.format(id=teos_id, n=n), headers={"User-Agent": "VUTT-HTR/1.0 (Tartu Ülikool)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("ei ole JPEG")
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(p)
    time.sleep(0.3)                                  # viisakus DWDS-i serveri vastu
    return p


def main():
    if "--naita" in sys.argv:
        i = sys.argv.index("--naita")
        t, pohjus = loe_teos(sys.argv[i + 1])[int(sys.argv[i + 2])]
        print(f"[{pohjus or 'OK'}]\n{t}")
        return
    if OUT.exists() and not DRY:
        sys.exit(f"{OUT} on juba olemas")
    rows, stats = [], Counter()
    for teos in TEOSED:
        lehed = loe_teos(teos)
        c = Counter(p or "OK" for _, p in lehed.values())
        print(f"{teos}: {len(lehed)} lk  {dict(c)}", flush=True)
        stats.update(c)
        for n, (t, pohjus) in sorted(lehed.items()):
            if pohjus:
                continue
            name = f"images/dta_{teos}_{n:04d}.jpg"
            if not DRY:
                try:
                    src = lae_pilt(teos, n)
                except Exception as e:
                    print(f"  pilt {teos} {n}: {e}", flush=True)
                    stats["pildi_viga"] += 1
                    continue
                (OUT / "images").mkdir(parents=True, exist_ok=True)
                os.link(src, OUT / name)
            rows.append((name, t, TEOSED[teos], teos))
    print(f"kokku: {dict(stats)} → {len(rows)} lehte")
    if DRY:
        return
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["failinimi", "transkriptsioon", "allikas"])
        w.writerows(r[:3] for r in rows)
    with open(OUT / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["failinimi", "projekt"])
        w.writerows((r[0], "dta:" + r[3]) for r in rows)
    (OUT / "SOURCE.txt").write_text(
        f"ehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_dta_kosmos.py\n"
        f"Deutsches Textarchiv, {KIRJELDUS}, CC BY 4.0\n"
        f"teosed: {', '.join(TEOSED)}\nlehti: {len(rows)}  välja: {dict(stats)}\n", encoding="utf-8")
    print(f"valmis: {OUT}")


if __name__ == "__main__":
    main()
