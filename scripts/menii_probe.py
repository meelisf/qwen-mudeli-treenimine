#!/usr/bin/env python3
"""
Menii ääreveeru sond: transformers (treeninguga sama ahel) vs GGUF

Küsimus, mida see lahendab: kui `print-base-r64-20260827` jätab `1635-1`
lehtedel `<m>` panemata, kas see on **mudeli** omadus või **ahela** oma?

GGUF-i tee saadab LANCZOS-PNG `fit_to_grid()`-võrel. Treening nägi hoopis
toorpilti, mille image processor skaleeris igal epohhil ise BICUBIC-uga
(ülevaade `docs/treening-ja-inferentsi-koodi-ulevaade-20260828.md`, punkt 5).
See skript käib **treeninguga samas ahelas**: toorpilt otse protsessorile.

  venv/bin/python scripts/menii_probe.py <mudel>

Väljund: data/vutt/reocr/menii-probe-<mudel>/
"""
import os, sys, time
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image as PILImage
from imaging import MAX_PIXELS
from prompt import INSTRUCTION

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-ocr-print-base-r64-20260827"
# 10 lehte, kus GGUF kaotas kogu veeru + 3 kontrolllehte, kus ei kaotanud
KADUNUD = ["0027","0029","0030","0020","0048","0043","0026","0031","0025","0037"]
KONTROLL = ["0028","0038","0024"]
VUTT = Path.home()/"vutt-backups/latest/data"
out = Path(f"data/vutt/reocr/menii-probe-{Path(MODEL).name}")
out.mkdir(parents=True, exist_ok=True)

pildid = {}
for n in KADUNUD + KONTROLL:
    hits = list(VUTT.rglob(f"r_acad_dorp_1635_1_{n}.jpg"))
    if hits:
        pildid[n] = hits[0]
print(f"Mudel: {MODEL}\nLehti: {len(pildid)}\nVäljund: {out}\n")

import torch
from unsloth import FastVisionModel
model, tok = FastVisionModel.from_pretrained(MODEL, load_in_4bit=False, dtype=torch.bfloat16)
tok.image_processor.size = {"longest_edge": MAX_PIXELS,
                            "shortest_edge": tok.image_processor.size.get("shortest_edge", 65536)}
FastVisionModel.for_inference(model)

chat = tok.apply_chat_template(
    [{"role":"user","content":[{"type":"text","text":INSTRUCTION},{"type":"image"}]}],
    add_generation_prompt=True, enable_thinking=False)

import re
t0 = time.time()
print(f"{'leht':6s} {'<m>':>4s} {'<i>':>4s} {'märke':>6s}  kiht")
for n in KADUNUD + KONTROLL:
    if n not in pildid: continue
    im = PILImage.open(pildid[n]).convert("RGB")     # TOORPILT, nagu treeningul
    inp = tok([im], [chat], add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=4096, do_sample=False, use_cache=True)
    t = tok.batch_decode(o, skip_special_tokens=True)[0]
    for mk in ["<|im_start|>assistant", "assistant\n"]:
        if mk in t: t = t.split(mk, 1)[-1]
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    (out/f"r_acad_dorp_1635_1_{n}.txt").write_text(t, encoding="utf-8")
    print(f"{n:6s} {t.count('<m>'):4d} {t.count('<i>'):4d} {len(t):6d}  "
          f"{'KADUNUD' if n in KADUNUD else 'kontroll'}", flush=True)
    im.close(); del inp, o; torch.cuda.empty_cache()
print(f"\nAeg: {(time.time()-t0)/60:.1f} min")
