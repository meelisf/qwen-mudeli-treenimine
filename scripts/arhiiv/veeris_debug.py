# Regioonide geomeetria valitud lehtedele (pildinime eesliite järgi)
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
import xml.etree.ElementTree as ET
soov = {a for a in sys.argv[1:] if not a.startswith('--')}
kirjed = json.load(open(vt.OUT / "veerisega_lehed.json", encoding="utf-8"))
gt = {}
import csv
for r in csv.DictReader(open(vt.GT_CSV, encoding="utf-8")):
    nr = Path(r["failinimi"]).name.split("_")[0]
    if nr in soov: gt[vt.norm(r["transkriptsioon"])[:400]] = Path(r["failinimi"]).name
leitud = set()
for allikas, stem, xs in vt.xmlid():
    a = vt.analyysi(xs)
    if not a: continue
    f = gt.get(vt.norm(a["tekst"])[:400])
    if not f or f in leitud: continue
    leitud.add(f)
    root = ET.fromstring(xs); ns = vt.ns_of(root); pg = root.find(f"{ns}Page")
    print(f"\n== {f[:40]}  page {pg.get('imageWidth')}x{pg.get('imageHeight')}")
    for r in a["regioonid"]:
        x0, y0, x1, y1 = r["bbox"]
        print(f"  {r['klass']:13s} {r['tyyp'][:10]:10s} x{x0:5d}-{x1:5d} y{y0:5d}-{y1:5d} rd{len(r['read']):3d}  {r['read'][0][:40]!r}")
        for tt, kk in r.get("veeris_sees", []):
            print(f"      VEERIS SEES ({kk}): {tt[:45]!r}")
        if r["klass"] in ("põhi", "teine_plokk") and "--read" in sys.argv:
            # reataseme x-algused: veeris põhiregiooni sees näeb välja nagu kõrvalekalle vasakule/paremale
            for t, b in r.get("ridade_bbox", []):
                if b: print(f"      rida x{b[0]:5d}-{b[2]:5d} y{b[1]:5d}  {t[:45]!r}")
