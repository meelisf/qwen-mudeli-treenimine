#!/usr/bin/env python3
"""
Fraktuurisond: mida maksis sildita Beckeri väljajätmine?

`print-base-r64-mi-vl-20260828` treeningust jäi välja 131 sildita Beckeri
lehte (`--valitud-lehekyljed`). Fraktuur kukkus 138 `⸗`-lehelt 72-le. See
skript mõõdab, kas fraktuuri transkriptsioon läks sellest alla.

**Kallutatus, mida tuleb tulemust lugedes arvestada.** Lehed 9–140 olid VANA
mudeli (`print-base-r64-20260827`) treeningus, uue omas mitte. Vanal on seega
mälueelis, mitte oskuseelis:

  uus samal tasemel  -> 131 lehe kaotus ei maksnud midagi
  uus veidi kehvem   -> osa vahest on vana mudeli päheõppimine
  uus selgelt kehvem -> kaotus on päris

Lehtedel 1–8 on VUTT-i GT olemas, aga need on MÕLEMA mudeli treeningus (vanal
koguni kahel kujul korraga: sildita `data/lehekyljed`-ist ja märgendatuna
VUTT-ist). CER-i sealt ei saa lugeda; `<cs>` paigutust saab.

Ahel on **treeninguga sama** – toorpilt otse protsessorile, nagu
`menii_probe.py`-s. Mitte GGUF, mitte `fit_to_grid`.

  venv/bin/python scripts/becker_probe.py                       # vaikimisi vana vs uus
  venv/bin/python scripts/becker_probe.py --n 40                # rohkem GT-ta lehti
  venv/bin/python scripts/becker_probe.py mudel_a mudel_b       # muud mudelid

NB! Nõuab vaba GPU-d (bf16, ~18 GB mudeli kohta). Treeningu ajal ei jookse.
Väljund: data/vutt/reocr/becker-probe-<mudel>/ + kokkuvõttetabel.
"""
import os
import re
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent))

import csv

from PIL import Image as PILImage

from imaging import MAX_PIXELS
from loop_detect import is_looped
from prompt import INSTRUCTION
from textmetrics import cer, strip_output, strip_tags

csv.field_size_limit(10 ** 9)

VAIKE_MUDELID = [
    "models/qwen3.5-ocr-print-base-r64-20260827",       # kontrollrühm, tootmises
    "models/qwen3.5-ocr-print-base-r64-mi-vl-20260828",  # uus
]
PILDID = Path("data/lehekyljed/images")
VUTT_CSV = Path("data/vutt/metadata.csv")

# ---------------------------------------------------------------------------
# Argumendid
# ---------------------------------------------------------------------------

argv = sys.argv[1:]
DRY = "--dry" in argv
if DRY:
    argv.remove("--dry")
N_GT_TA = 20
if "--n" in argv:
    i = argv.index("--n")
    N_GT_TA = int(argv[i + 1])
    del argv[i:i + 2]
mudelid = argv or VAIKE_MUDELID

# Lehed 1–8: VUTT-i GT olemas (mõlema mudeli treeningus – vt docstring).
# Lehed 9–140: GT-d EI OLE. Ühtlaselt hajutatud valik, et võrdlus oleks
# jooksude vahel korratav – juhuslikku seemet siin meelega ei ole.
GT_LEHED = [f"{n:05d}" for n in range(1, 9)]
koik_gt_ta = [n for n in range(9, 141) if (PILDID / f"{n:05d}.jpg").exists()]
if N_GT_TA >= len(koik_gt_ta):
    valik = koik_gt_ta
else:
    # Ühtlane hajutus ÜLE KOGU vahemiku, mõlemad otsad kaasa arvatud. Lihtne
    # `[::samm]` jätaks saba katmata (20 lehe puhul lõppes valik lk 123 peal).
    valik = [koik_gt_ta[round(i * (len(koik_gt_ta) - 1) / (N_GT_TA - 1))]
             for i in range(N_GT_TA)] if N_GT_TA > 1 else koik_gt_ta[:1]
GT_TA_LEHED = [f"{n:05d}" for n in dict.fromkeys(valik)]
KOIK = GT_LEHED + GT_TA_LEHED

# ---------------------------------------------------------------------------
# VUTT-i GT lehtedele 1–8
# ---------------------------------------------------------------------------

GT = {}
if VUTT_CSV.exists():
    for rida in csv.DictReader(open(VUTT_CSV, encoding="utf-8")):
        b = os.path.basename(rida["failinimi"])
        m = re.match(r"1644-becker-linteum__(\d+)\.jpg$", b)
        if m:
            GT[m.group(1)] = rida["transkriptsioon"] or ""

puuduvad = [n for n in KOIK if not (PILDID / f"{n}.jpg").exists()]
print(f"Mudelid:   {len(mudelid)}")
for m in mudelid:
    print(f"           {m}")
print(f"Lehti:     {len(KOIK)}  ({len(GT_LEHED)} GT-ga, {len(GT_TA_LEHED)} GT-ta)")
print(f"GT VUTT-ist: {len(GT)} lehte")
if puuduvad:
    print(f"HOIATUS: {len(puuduvad)} pilti puudub: {puuduvad}")
print()

# ---------------------------------------------------------------------------
# Genereerimine – üks mudel korraga, teine alles pärast vabastamist
# ---------------------------------------------------------------------------


def jooksuta(mudel: str) -> dict:
    """Transkribeerib kõik lehed. Olemasolevat väljundit ei genereeri uuesti."""
    out = Path(f"data/vutt/reocr/becker-probe-{Path(mudel).name}")
    out.mkdir(parents=True, exist_ok=True)

    tulemus, puudu = {}, []
    for n in KOIK:
        f = out / f"{n}.txt"
        if f.exists():
            tulemus[n] = f.read_text(encoding="utf-8")
        elif (PILDID / f"{n}.jpg").exists():
            puudu.append(n)
    if not puudu:
        print(f"{mudel}: kõik {len(tulemus)} lehte on juba kettal, ei genereeri.")
        return tulemus

    import torch
    from unsloth import FastVisionModel

    print(f"\n=== {mudel} — genereerin {len(puudu)} lehte ===")
    model, tok = FastVisionModel.from_pretrained(
        mudel, load_in_4bit=False, dtype=torch.bfloat16)
    tok.image_processor.size = {
        "longest_edge": MAX_PIXELS,
        "shortest_edge": tok.image_processor.size.get("shortest_edge", 65536)}
    FastVisionModel.for_inference(model)

    chat = tok.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION}, {"type": "image"}]}],
        add_generation_prompt=True, enable_thinking=False)

    t0 = time.time()
    print(f"{'leht':6s} {'<cs>':>5s} {'⸗':>4s} {'<m>':>4s} {'märke':>6s}  märkus")
    for n in puudu:
        im = PILImage.open(PILDID / f"{n}.jpg").convert("RGB")   # TOORPILT
        inp = tok([im], [chat], add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=4096,
                               do_sample=False, use_cache=True)
        t = strip_output(tok.batch_decode(o, skip_special_tokens=True)[0])
        (out / f"{n}.txt").write_text(t, encoding="utf-8")
        tulemus[n] = t
        loop = is_looped(t)
        print(f"{n:6s} {t.count('<cs>'):5d} {t.count('⸗'):4d} {t.count('<m>'):4d} "
              f"{len(t):6d}  {'LOOP ' + str(loop) if loop else ''}"
              f"{' GT' if n in GT_LEHED else ''}", flush=True)
        im.close()
        del inp, o
        torch.cuda.empty_cache()
    print(f"Aeg: {(time.time() - t0) / 60:.1f} min")

    del model, tok
    torch.cuda.empty_cache()
    return tulemus


if DRY:
    print("--dry: GPU-d ei puututa, ainult plaan.\n")
    print(f"GT-ga lehed   ({len(GT_LEHED):2d}): {' '.join(GT_LEHED)}")
    print(f"GT-ta lehed   ({len(GT_TA_LEHED):2d}): {' '.join(GT_TA_LEHED)}")
    for m in mudelid:
        d = Path(f"data/vutt/reocr/becker-probe-{Path(m).name}")
        olemas = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"\n{m}")
        print(f"  mudel kettal: {'JAH' if Path(m).exists() else 'EI – tuleb enne treenida'}")
        print(f"  juba genereeritud: {olemas}/{len(KOIK)} lehte")
    print(f"\nHinnanguline aeg: ~{len(KOIK) * len(mudelid) * 30 / 60:.0f} min "
          f"({len(KOIK)} lk x {len(mudelid)} mudelit x ~30 s)")
    sys.exit(0)

valjundid = {m: jooksuta(m) for m in mudelid}

# ---------------------------------------------------------------------------
# Kokkuvõte
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("1) LEHED 1–8 — VUTT-i GT vastu")
print("   NB! Mõlema mudeli treeningus. CER ei ole aus võrdlus; `<cs>` on")
print("   siiski huvitav: uus mudel õppis neil lehtedel just koodivahetust.")
print("=" * 78)
print(f"{'mudel':46s} {'cer_plain':>10s} {'<cs> GT':>8s} {'<cs> väljund':>13s}")
for m in mudelid:
    v = valjundid[m]
    cerid, cs_gt, cs_out = [], 0, 0
    for n in GT_LEHED:
        if n not in v or n not in GT:
            continue
        cerid.append(cer(strip_tags(GT[n]), strip_tags(v[n])))
        cs_gt += GT[n].count("<cs>")
        cs_out += v[n].count("<cs>")
    keskm = sum(cerid) / len(cerid) if cerid else float("nan")
    print(f"{Path(m).name:46s} {keskm:9.1%} {cs_gt:8d} {cs_out:13d}")

print("\n" + "=" * 78)
print(f"2) LEHED 9–140 ({len(GT_TA_LEHED)} tk) — GT-d EI OLE")
print("   Vana mudel NÄGI neid treeningus, uus mitte. Vt skripti docstring'i.")
print("=" * 78)
print(f"{'mudel':46s} {'märke/lk':>9s} {'<cs>':>6s} {'⸗':>6s} {'<m>':>5s} {'loope':>6s}")
for m in mudelid:
    v = valjundid[m]
    lehed = [n for n in GT_TA_LEHED if n in v]
    if not lehed:
        continue
    print(f"{Path(m).name:46s} "
          f"{sum(len(v[n]) for n in lehed) / len(lehed):9.0f} "
          f"{sum(v[n].count('<cs>') for n in lehed):6d} "
          f"{sum(v[n].count('⸗') for n in lehed):6d} "
          f"{sum(v[n].count('<m>') for n in lehed):5d} "
          f"{sum(1 for n in lehed if is_looped(v[n])):6d}")

if len(mudelid) == 2:
    a, b = mudelid
    va, vb = valjundid[a], valjundid[b]
    ühised = [n for n in GT_TA_LEHED if n in va and n in vb]
    if ühised:
        erinevused = sorted(
            ((cer(strip_tags(va[n]), strip_tags(vb[n])), n) for n in ühised),
            reverse=True)
        keskm = sum(c for c, _ in erinevused) / len(erinevused)
        print(f"\n   Mudelite lahknevus omavahel (CER, tagideta): keskm {keskm:.1%}, "
              f"mediaan {erinevused[len(erinevused) // 2][0]:.1%}")
        print("   Suurima lahknevusega lehed – need vaata silmaga üle:")
        for c, n in erinevused[:5]:
            print(f"     {n}  {c:6.1%}   "
                  f"vana {len(va[n]):5d} mrk / uus {len(vb[n]):5d} mrk")

print("\nVäljundid: data/vutt/reocr/becker-probe-<mudel>/")
print("GT-d lehtedele 9–140 ei ole. Kui tahad päris mõõtu, märgi VUTT-is paar")
print("neist lehtedest valmis JA hoia treeningust väljas (holdout.txt).")
