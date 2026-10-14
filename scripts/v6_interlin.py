"""Interlineaarsete lisanduste kandidaadid PAGE XML-is (analüüs, mitte ehitus).

Kandidaat L: lühike rida (laius < 0,5 regiooni rea mediaanist), mille baasjoon jääb kahe
JÄRJESTIKUSE põhirea A ja B vahele (B−A < 1,4 × reasamm) ja mis kattub x-is mõlemaga.
"""
import csv, re, sys, statistics as st, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, "scripts")
import veeris_tuvastus as vt

RANGE = "--range" in sys.argv
V6 = Path("data/kurrent_v6/metadata.csv")

def baasjoon(line, ns):
    b = line.find(f"{ns}Baseline")
    if b is not None and b.get("points"):
        ys = [float(p.split(",")[1]) for p in b.get("points").split()]
        return sum(ys) / len(ys)
    bb = vt.bbox(line, ns)
    return bb[3] if bb else None

def read(r, ns):
    out = []
    for line in r.findall(f"{ns}TextLine"):
        te = line.find(f"{ns}TextEquiv"); uc = te.find(f"{ns}Unicode") if te is not None else None
        bb = vt.bbox(line, ns); y = baasjoon(line, ns)
        if uc is not None and uc.text and uc.text.strip() and bb and y is not None:
            out.append(dict(t=uc.text.strip(), x0=bb[0], x1=bb[2], y=y))
    return out

def kandidaadid(xs):
    try:
        root = ET.fromstring(xs)
    except ET.ParseError:
        return None
    ns = vt.ns_of(root); res = []
    for r in root.iter(f"{ns}TextRegion"):
        L = read(r, ns)
        if len(L) < 4:
            continue
        mw = st.median(l["x1"] - l["x0"] for l in L)
        pikad = sorted([l for l in L if l["x1"] - l["x0"] >= 0.6 * mw], key=lambda l: l["y"])
        if len(pikad) < 3:
            continue
        sammud = [b["y"] - a["y"] for a, b in zip(pikad, pikad[1:]) if b["y"] > a["y"]]
        if not sammud:
            continue
        samm = st.median(sammud)
        for l in L:
            w = l["x1"] - l["x0"]
            if w >= 0.5 * mw:
                continue
            yle = [p for p in pikad if p["y"] < l["y"] - 0.15 * samm and min(p["x1"], l["x1"]) - max(p["x0"], l["x0"]) > 0.5 * w]
            alla = [p for p in pikad if p["y"] > l["y"] + 0.15 * samm and min(p["x1"], l["x1"]) - max(p["x0"], l["x0"]) > 0.5 * w]
            if yle and alla:
                a, b = yle[-1], alla[0]
                tiht = b["y"] - a["y"] < 1.2 * samm
                vasak = abs(l["x0"] - a["x0"]) > 0.05 * mw and abs(l["x0"] - b["x0"]) > 0.05 * mw
                if b["y"] - a["y"] < 1.4 * samm and (not RANGE or (tiht and vasak)):
                    i = {id(x): n for n, x in enumerate(L)}
                    jrk = "A-L-B" if i[id(a)] < i[id(l)] < i[id(b)] else ("L enne A" if i[id(l)] < i[id(a)] else ("L pärast B" if i[id(l)] > i[id(b)] else "muu"))
                    res.append((a["t"], l["t"], b["t"], jrk, (l["x0"], int(l["y"] - samm), l["x1"], int(b["y"] + 0.3 * samm), a["x0"], a["x1"])))
    return res

v6 = defaultdict(set)
for r in csv.DictReader(open(V6, encoding="utf-8")):
    v6[r["allikas"]].add(vt.norm(r["transkriptsioon"])[:400])
import random; random.seed(1); JRK = Counter(); POOL = []
lehti = Counter(); ridu = Counter(); kokku = Counter(); naited = defaultdict(list)
for allikas, stem, xs in vt.xmlid():
    a = vt.analyysi(xs)
    if not a or vt.norm(a["tekst"])[:400] not in v6[allikas]:
        continue
    kokku[allikas] += 1
    k = kandidaadid(xs)
    if k:
        lehti[allikas] += 1; ridu[allikas] += len(k)
        for x in k: JRK[x[3]] += 1
        POOL.append((allikas, stem, a["tekst"], k))
        if len(naited[allikas]) < 6:
            naited[allikas].append((stem, k[0]))
print(f"{'allikas':26s} {'v6 lehti':>8s} {'kand.lehti':>10s} {'kand.ridu':>9s}")
for al in sorted(kokku, key=lambda a: -lehti[a]):
    print(f"{al:26s} {kokku[al]:8d} {lehti[al]:10d} {ridu[al]:9d}")
print("KOKKU", sum(kokku.values()), sum(lehti.values()), sum(ridu.values()))
for al, ns_ in naited.items():
    print("\n==", al)
    for stem, (a, l, b, *_ ) in ns_:
        print(f"  {stem[:40]}\n     A: {a[:70]}\n   + L: {l}\n     B: {b[:70]}")

print("XML-järjekord:", dict(JRK))
import json; json.dump(random.sample(POOL, min(24, len(POOL))), open("/tmp/interlin_sample%s.json" % ("_r" if RANGE else ""), "w"), ensure_ascii=False)
