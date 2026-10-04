#!/usr/bin/env python3
"""
Üks VUTT-i leht (Oesel m3do2t lk 3) mitme llama-serveri vastu + CER/WER.

  python scripts/oesel_proov.py vana=http://127.0.0.1:8091 uus=http://127.0.0.1:8092

Leht on telefonifoto, mitte skaneering, ja ei ole üheski treeningkomplektis —
see on lähim, mis meil on „päris VUTT-i lehele" (holdout ei sisalda VUTT-i
žanre, SEIS §3.4). Võrdlus 21 HTR-mudeliga: VUTT repo
`docs/reviews/2026-10-02-htr-mudelite-vordlus.md`, kus kurrent-20260829 sai
CER norm 2,6 % / toores 5,3 % / WER 13,3 %.

Päring on sama mis `eval_kurrent.py`-s ja teenuses: `fit_to_grid` + PNG,
`KURRENT_INSTRUCTION`, temperature 0, mõtlemine väljas. Tulemus on
deterministlik — `vana` peab andma sama teksti mis 02.10 jooks
(`viide_20261002_vana.txt`); kui ei anna, on tingimused nihkunud ja võrdlus
ei kehti.

Mõõdikud on VUTT-i `score.py`-ga samad (et numbrid oleksid tabeliga
võrreldavad): toores = tühikud ühtlustatud; norm = ſ=s, poolitusmärgid
ühtseks, arhiivitempel välja, reavahetus = tühik; WER normaliseeritud tekstil.
"""

import base64
import io
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import editdistance
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from imaging import fit_to_grid
from prompt import KURRENT_INSTRUCTION
from textmetrics import strip_output

KAUST = Path("data/kurrent/oesel")
PILT = KAUST / "m3do2t_lk003.jpeg"
GT_PATH = KAUST / "m3do2t_lk003.txt"
VIIDE = KAUST / "viide_20261002_vana.txt"   # kurrent-20260829 Q8_0, 02.10
OUT = KAUST / "valjundid"


def raw(t):
    t = unicodedata.normalize("NFC", t)
    return "\n".join(" ".join(l.split()) for l in t.strip().splitlines() if l.strip())


def norm(t):
    t = raw(t)
    t = "\n".join(l for l in t.splitlines() if not re.search(r"R\.?\s*19\.?\s*G", l))
    t = t.replace("ſ", "s").replace("⸗", "-").replace("¬", "-").replace("=", "-")
    return " ".join(t.split())


def cer(h, g, f):
    h, g = f(h), f(g)
    return editdistance.eval(h, g) / len(g)


def wer(h, g):
    h, g = norm(h).split(), norm(g).split()
    return editdistance.eval(h, g) / len(g)


def saada(endpoint: str) -> str:
    with PILImage.open(PILT) as im:
        pilt = fit_to_grid(im.convert("RGB"))
        buf = io.BytesIO()
        pilt.save(buf, "PNG", optimize=False)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    keha = json.dumps({
        "model": "kurrent",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": KURRENT_INSTRUCTION},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}],
        "max_tokens": 4096,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/chat/completions",
                                 data=keha, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return strip_output(json.loads(r.read())["choices"][0]["message"]["content"])


paarid = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
if not paarid:
    print(__doc__)
    sys.exit(1)

GT = GT_PATH.read_text(encoding="utf-8")
OUT.mkdir(parents=True, exist_ok=True)

valjundid = {}
if VIIDE.exists():
    valjundid["viide 02.10 (vana)"] = VIIDE.read_text(encoding="utf-8")
for nimi, endpoint in paarid:
    tekst = saada(endpoint)
    (OUT / f"{nimi}.txt").write_text(tekst, encoding="utf-8")
    valjundid[nimi] = tekst

print("\n| mudel | CER norm | CER toores | WER |")
print("|---|---|---|---|")
for nimi, tekst in valjundid.items():
    numbrid = f"{cer(tekst, GT, norm):.1%} | {cer(tekst, GT, raw):.1%} | {wer(tekst, GT):.1%}"
    print(f"| {nimi} | {numbrid.replace('.', ',')} |")

if "vana" in valjundid and "viide 02.10 (vana)" in valjundid:
    sama = raw(valjundid["vana"]) == raw(valjundid["viide 02.10 (vana)"])
    print(f"\nVana mudel == 02.10 väljund: {'JAH' if sama else 'EI — tingimused nihkunud, kontrolli!'}")

# Teadaolevad korduvad nimevead (htr-võrdlus 02.10): kas uus parandas?
for vale, oige in [("Osel", "Oesel"), ("Eresparre", "Ekesparre")]:
    muster = re.compile(r"\b" + vale + r"\b")
    rida = ", ".join(f"{n}: {len(muster.findall(t))}× {vale}" for n, t in valjundid.items())
    print(f"{oige}: {rida}")
print(f"\nVäljundid: {OUT}/")
