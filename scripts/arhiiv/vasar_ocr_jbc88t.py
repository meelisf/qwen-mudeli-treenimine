#!/usr/bin/env python3
"""jbc88t (Academia Gustaviana põhikiri, Baumanni ärakiri) OCR v5-ga Vasari klapituse jaoks (14.10).

Kasutab tootmise käsikirjaserverit (8081, kurrent-20261013) sama päringuga mis reocr_vutt.py
(fit_to_grid + PNG, KURRENT_INSTRUCTION_V5, temperature 0). VUTT-i EI kirjuta.
  venv/bin/python scripts/vasar_ocr_jbc88t.py → data/vasar/ocr_v5/<tüvi>.txt
"""
import base64, io, json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, "scripts")
from imaging import fit_to_grid               # noqa: E402
from loop_detect import is_looped             # noqa: E402
from prompt import KURRENT_INSTRUCTION_V5     # noqa: E402

ENDPOINT = "http://127.0.0.1:8081"


def png_uri(path):
    # sama mis reocr_vutt.pildi_data_uri: fit_to_grid + PNG (mitte JPEG)
    with PILImage.open(path) as im:
        buf = io.BytesIO(); fit_to_grid(im.convert("RGB")).save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saada(img):
    keha = json.dumps({"model": "vutt", "messages": [{"role": "user", "content": [
        {"type": "text", "text": KURRENT_INSTRUCTION_V5},
        {"type": "image_url", "image_url": {"url": png_uri(img)}}]}],
        "max_tokens": 4096, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{ENDPOINT}/v1/chat/completions", data=keha,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        v = json.loads(r.read())
    t = re.sub(r"<think>.*?</think>", "", v["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
    t = re.sub(r"^```[a-z]*\n?|\n?```$", "", t).strip()
    return t, v["choices"][0].get("finish_reason")


SIS = Path("data/vasar/jbc88t"); OUT = Path("data/vasar/ocr_v5"); OUT.mkdir(exist_ok=True)


def too(img):
    o = OUT / (img.stem + ".txt")
    if o.exists():
        return img.stem, "olemas"
    t, fr = saada(img)
    o.write_text(t + "\n", encoding="utf-8")
    return img.stem, f"{fr} {len(t.splitlines())} rida{' LOOP' if is_looped(t) else ''}"


with ThreadPoolExecutor(2) as ex:
    for s, info in ex.map(too, sorted(SIS.glob("*.jpg"))):
        print(s[-10:], info, flush=True)
