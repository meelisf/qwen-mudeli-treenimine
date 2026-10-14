import csv, json, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, "scripts"); import veeris_tuvastus as vt
idx = {}
for r in csv.DictReader(open("data/kurrent_v6/metadata.csv", encoding="utf-8")):
    idx[(r["allikas"], vt.norm(r["transkriptsioon"])[:400])] = r["failinimi"]
out = Path("data/v6_veeris/interlin"); out.mkdir(exist_ok=True)
for n, (al, stem, tekst, k) in enumerate(json.load(open("/tmp/interlin_sample.json"))):
    f = idx.get((al, vt.norm(tekst)[:400]))
    if not f: continue
    a, l, b, jrk, (x0, y0, x1, y1, ax0, ax1) = k[0]
    im = Image.open("data/kurrent_v6/" + f).convert("RGB")
    d = ImageDraw.Draw(im); d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
    pad = (y1 - y0)
    c = im.crop((max(0, ax0 - 30), max(0, y0 - pad), min(im.width, ax1 + 30), min(im.height, y1 + pad)))
    c.thumbnail((1400, 500)); c.save(out / f"{n:02d}_{al[:12]}.jpg", quality=80)
    print(f"{n:02d} {al} {f} | L={l!r} | {jrk}")
