#!/usr/bin/env python3
"""Ladinakeelsete käsikirja-allikate PAGE XML audit (Leibniz, Gwalther).

Sama mõõdupuu mis `tuhjad_read_audit.py`-l: tühjad TextLine'id (GT-st
puuduvad read, mis pildil on), looja (Creator — Matcher = automaatjoondus),
regioonide tüübid (marginaalid), keel funktsioonisõnade järgi. Lisaks
pildi olemasolu ja kustutatud teksti kahtlus (kahekordsed sõnad
„De De", „vehe vehemens" — Leibnizi GT kirjutab maha tõmmatud teksti
märgendita välja).

Väljund: data/kurrent_xix_audit/ladina_audit.csv + kokkuvõte stdout-i.
Käivitus: venv/bin/python scripts/audit_ladina.py
"""
import csv, re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw_xml")
OUT = Path("data/kurrent_xix_audit/ladina_audit.csv")
ALLIKAD = {
    "leibniz_clean": sorted((RAW / "leibniz/train/clean").glob("*.xml")),
    "leibniz_val": sorted((RAW / "leibniz/val").glob("*.xml")),
    "gwalther": sorted(RAW.glob("gwalther/*/page/*.xml")),
}
LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)
CREATOR = re.compile(r"<Creator>(.*?)</Creator>", re.S)
REGION = re.compile(r"<TextRegion\b[^>]*?(?:type=\"(\w+)\"|structure \{type:(\w+);)", re.S)
SW = {
    "la": set("et in est ad cum non quod qui quae sed ut per ex de ac atque eius esse sunt vel nec enim ab hoc etiam quam autem pro".split()),
    "fr": set("le la les et de des du que qui est pour dans une un pas par sur au aux vous nous ce il".split()),
    "de": set("und der die das den dem des ist nicht ein eine zu mit von auf sich auch als wie daß so wird".split()),
}


def rea_tekst(tl):
    u = UNI.findall(WORD.sub("", tl))        # rea enda TextEquiv, mitte sõnade oma
    return u[-1].strip() if u else ""


def looja(xml):
    m = CREATOR.search(xml)
    c = m.group(1) if m else ""
    if "Matcher" in c:
        return "matcher"
    return "transkribus" if "Transkribus" in c else (c.split(":")[0][:40] or "?")


def keel(tekst):
    w = re.findall(r"[a-zàâçéèêëîïôûùüÿœæäöß]+", tekst.lower())
    sc = {k: sum(1 for x in w if x in s) for k, s in SW.items()}
    return max(sc, key=sc.get) if sum(sc.values()) >= 3 else "?"


def kordused(read):
    # kahekordne sõna või sõnaalgus järgneva sõna ees (vehe vehemens) — maha tõmmatud teksti kahtlus
    n = 0
    for r in read:
        w = r.split()
        n += sum(1 for a, b in zip(w, w[1:]) if len(a) >= 2 and b.lower().startswith(a.lower().rstrip(".,;:")))
    return n


def pilt(xml_path):
    for ext in (".jpg", ".jpeg", ".png", ".tif"):
        for p in (xml_path.with_suffix(ext), xml_path.parent.parent / "img" / (xml_path.stem + ext)):
            if p.exists():
                return p
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for allikas, files in ALLIKAD.items():
        for f in files:
            x = f.read_text(encoding="utf-8")
            lines = LINE.findall(x)
            read = [rea_tekst(tl) for tl in lines]
            tekst = "\n".join(r for r in read if r)
            regs = Counter(a or b for a, b in REGION.findall(x))
            rows.append({
                "allikas": allikas, "fail": f.name, "pilt": "jah" if pilt(f) else "EI",
                "looja": looja(x), "ridu": len(read), "tuhje": sum(1 for r in read if not r),
                "marginaal_reg": sum(v for k, v in regs.items() if "Margin" in k or k == "marginalia"),
                "regioonid": ";".join(f"{k}:{v}" for k, v in regs.most_common()),
                "keel": keel(tekst), "kordusi": kordused(read), "tahti": len(tekst),
            })
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for allikas in ALLIKAD:
        r = [x for x in rows if x["allikas"] == allikas]
        if not r:
            print(allikas, "— faile ei leitud")
            continue
        print(f"\n== {allikas}: {len(r)} lehte, {sum(x['ridu'] for x in r)} rida, {sum(x['tahti'] for x in r)} tähte")
        print("  pilt puudu:", sum(1 for x in r if x["pilt"] == "EI"))
        print("  looja:", dict(Counter(x["looja"] for x in r)))
        print("  keel:", dict(Counter(x["keel"] for x in r)))
        t = Counter(min(x["tuhje"], 2) for x in r)
        print(f"  tühje ridu: 0 → {t[0]}, 1 → {t[1]}, ≥2 → {t[2]}")
        print("  marginaaliregiooniga lehti:", sum(1 for x in r if x["marginaal_reg"]))
        print("  kordusega lehti (kustutuse kahtlus):", sum(1 for x in r if x["kordusi"]),
              "kordusi kokku", sum(x["kordusi"] for x in r))
        regs = Counter()
        for x in r:
            for kv in filter(None, x["regioonid"].split(";")):
                k, v = kv.rsplit(":", 1)
                regs[k] += int(v)
        print("  regioonitüübid:", dict(regs.most_common()))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
