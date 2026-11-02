#!/usr/bin/env python3
"""Kontrollnimekiri veerise tuvastusele: lipuga lehed + juhuvalim (3/allikas), pildid kastidega.

  venv/bin/python scripts/veeris_kontroll.py   → data/v6_veeris/KONTROLL.md + kontroll/*.jpg
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt

OUT = vt.OUT
random.seed(14102026)
kirjed = [x for x in json.load(open(OUT / "veerisega_lehed.json", encoding="utf-8")) if x["pilt"]]
lipuga = [x for x in kirjed if x["lipud"]]
# ainult_numbrid on sage (ploki kõrval olevad numbrid = veeris, kasutaja 14.10) → kontrolliks 2/allikas
_num = defaultdict(list)
for x in lipuga:
    if x["lipud"] == ["ainult_numbrid"]:
        _num[x["allikas"]].append(x)
_num_kogus = {a: len(v) for a, v in _num.items()}
lipuga = [x for x in lipuga if x["lipud"] != ["ainult_numbrid"]] + \
         [y for a in sorted(_num) for y in random.Random(1).sample(_num[a], min(2, len(_num[a])))]
per = defaultdict(list)
for x in kirjed:
    if not x["lipud"]:
        per[x["allikas"]].append(x)
valim = [y for a in sorted(per) for y in random.sample(per[a], min(3, len(per[a])))]
vaja = {(x["allikas"], x["xml"]) for x in lipuga + valim}

(OUT / "kontroll").mkdir(exist_ok=True)
vt.NAITEID = 0
nimi = {}
for allikas, stem, xs in vt.xmlid():
    if (allikas, stem) not in vaja or (allikas, stem) in nimi:
        continue
    a = vt.analyysi(xs)
    if a is None or not any(r["klass"].startswith("veeris") or r.get("veeris_sees") for r in a["regioonid"]):
        continue   # sama tüvi teises tarballis — mitte see leht
    fail = next(x["pilt"] for x in kirjed if x["allikas"] == allikas and x["xml"] == stem)
    pilt = Path(fail).name
    vt.OUT = OUT / "kontroll"
    p = vt.IMG / pilt
    if p.exists():
        from PIL import Image, ImageDraw
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
        varv = {"põhi": "blue", "number": "gray", "teine_plokk": "orange", "põhi_lisa": "green", "päis": "purple", "jalus": "purple", "naaberleht": "black"}
        for r in a["regioonid"]:
            d.rectangle(list(r["bbox"]), outline=varv.get(r["klass"], "red"), width=8)
            sees = {tt for tt, _ in r.get("veeris_sees", [])}
            for tt, bb in r.get("ridade_bbox", []):
                if bb and tt in sees:
                    d.rectangle(list(bb), outline="red", width=6)
        im.thumbnail((1800, 1800)); im.save(OUT / "kontroll" / pilt, quality=85)
    nimi[(allikas, stem)] = pilt
vt.OUT = OUT


def rida(x):
    v = "; ".join(" / ".join(r["read"][:2])[:70] for r in x["veeris"])
    return f"| `{nimi.get((x['allikas'], x['xml']), x['pilt'])}` | {', '.join(x['lipud']) or '–'} | {v} |"


with open(OUT / "KONTROLL.md", "w", encoding="utf-8") as f:
    f.write("# Veerise tuvastus — kontrollnimekiri\n\nPildid kastidega: `data/v6_veeris/kontroll/<nimi>` "
            "(sinine = põhitekst, oranž = teine plokk/lehekülg, punane = VEERIS, hall = number, roheline = põhi_lisa, lilla = päis/jalus, must = naaberlehe tekst).\n\n")
    f.write("## A. Juhuvalim (liputa) — kas punane on päris veeris?\n\n| pilt | lipud | veerise algus |\n|---|---|---|\n")
    f.write("\n".join(rida(x) for x in valim) + "\n\n")
    for lipp, selgitus in [("laotus_yhes", "kahe lehekülje laotus ÜHES regioonis — veeris paigutamatu, kas välja?"),
                           ("ainult_numbrid", "ainult numbrid — veerise summa/viide (jääb) või arhiivinumber (välja)?"),
                           ("lisamärk", "lisamärgiga — kas <add> (ADR 0067)?"),
                           ("pikk", "≥ 8 rida — kas veeris või hoopis veerg/lisatud lõik?")]:
        xs = [x for x in lipuga if lipp in x["lipud"]]
        if lipp == "ainult_numbrid":
            f.write(f"Kokku allikate kaupa: {_num_kogus} — allpool valim 2/allikas.\n\n")
        f.write(f"## B. `{lipp}` ({len(xs)}) — {selgitus}\n\n| pilt | lipud | veerise algus |\n|---|---|---|\n")
        f.write("\n".join(rida(x) for x in xs) + "\n\n")
print(len(valim), "valimis,", len(lipuga), "lipuga,", len(nimi), "pilti")
