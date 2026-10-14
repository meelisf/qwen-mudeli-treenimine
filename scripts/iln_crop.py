import json
from pathlib import Path
from PIL import Image, ImageDraw
out = Path("data/v6_veeris/interlin_add"); out.mkdir(exist_ok=True)
for n, (f, al, tekst, bbs) in enumerate(json.load(open("/tmp/iln_naited.json"))):
    im = Image.open("data/kurrent_v5m/" + f).convert("RGB"); d = ImageDraw.Draw(im)
    for j, bb in enumerate(bbs):
        d.rectangle(bb, outline=(0, 0, 255) if j == 0 else (255, 0, 0), width=3)
    x0 = min(b[0] for b in bbs); x1 = max(b[2] for b in bbs); y0 = min(b[1] for b in bbs); y1 = max(b[3] for b in bbs)
    h = y1 - y0
    c = im.crop((max(0, x0 - 40), max(0, y0 - h), min(im.width, x1 + 40), min(im.height, y1 + h // 2)))
    c.thumbnail((1400, 330)); c.save(out / f"{n:02d}.jpg", quality=80)
    print(f"{n:02d} {al[:14]:14s} {tekst}")
