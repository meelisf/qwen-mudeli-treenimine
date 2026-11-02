#!/usr/bin/env python3
"""
Menii ääreveeru sond GGUF-serveri vastu (tootmisega sama ahel)

`menii_probe.py` käib treeninguga samas ahelas (unsloth bf16, toorpilt).
Tootmine jookseb aga llama.cpp Q8_0 + `fit_to_grid` PNG — see skript mõõdab
SAMU 13 lehte selles ahelas, mitu serverit järjest (mitte paralleelselt,
muidu jagavad nad GPU-d).

  venv/bin/python scripts/menii_gguf.py vana=http://127.0.0.1:8091 uus=http://127.0.0.1:8092

Juhis tuleb `prompt.INSTRUCTION`-ist, mis on jooksu ajal kettal — kui kaks
mudelit vajavad eri juhist, jooksuta neid eraldi kutsetena.

Väljund: data/vutt/reocr/menii-gguf-<silt>/r_acad_dorp_1635_1_<n>.txt
"""
import base64, io, json, re, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image as PILImage
from imaging import fit_to_grid
from loop_detect import is_looped
from prompt import INSTRUCTION

# sama valik mis menii_probe.py-s
KADUNUD = ["0027","0029","0030","0020","0048","0043","0026","0031","0025","0037"]
KONTROLL = ["0028","0038","0024"]
VUTT = Path.home()/"vutt-backups/latest/data"

serverid = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
if not serverid:
    sys.exit("Kasutus: menii_gguf.py silt=http://127.0.0.1:8091 [silt2=...]")

pildid = {}
for n in KADUNUD + KONTROLL:
    hits = list(VUTT.rglob(f"r_acad_dorp_1635_1_{n}.jpg"))
    if hits:
        pildid[n] = hits[0]
print(f"Lehti: {len(pildid)}/13")


def data_uri(path: Path) -> str:
    # PNG + fit_to_grid, nagu reocr_vutt.py ja teenus (vt sealset docstringi)
    with PILImage.open(path) as im:
        buf = io.BytesIO()
        fit_to_grid(im.convert("RGB")).save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def saada(endpoint: str, img: Path):
    keha = json.dumps({
        "model": "vutt",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image_url", "image_url": {"url": data_uri(img)}}]}],
        "max_tokens": 4096,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{endpoint}/v1/chat/completions", data=keha,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        v = json.loads(r.read())
    t = v["choices"][0]["message"]["content"]
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    return t, v["choices"][0].get("finish_reason")


tulemus = {}
for silt, endpoint in serverid:
    out = Path(f"data/vutt/reocr/menii-gguf-{silt}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, p in pildid.items():
        t, fin = saada(endpoint.rstrip("/"), p)
        (out/f"r_acad_dorp_1635_1_{n}.txt").write_text(t, encoding="utf-8")
        tulemus[(silt, n)] = (t.count("<m>"), t.count("<i>"), len(t),
                              fin == "length", is_looped(t))
    print(f"{silt}: {(time.time()-t0)/60:.1f} min")

sildid = [s for s, _ in serverid]
print(f"\n{'leht':6s} " + " ".join(f"{s+' <m>':>10s} {'<i>':>4s} {'märke':>6s}" for s in sildid) + "  kiht")
for n in pildid:
    rida = []
    for s in sildid:
        m, i, c, cut, loop = tulemus[(s, n)]
        lipp = ("L" if loop else "") + ("K" if cut else "")
        rida.append(f"{m:10d} {i:4d} {c:6d}{lipp:2s}")
    print(f"{n:6s} " + " ".join(rida) + f"  {'KADUNUD' if n in KADUNUD else 'kontroll'}")
for s in sildid:
    mid = [tulemus[(s, n)] for n in pildid]
    print(f"{s}: <m> kokku {sum(x[0] for x in mid)}, lehti <m>-ga {sum(1 for x in mid if x[0])}/{len(mid)}, "
          f"loope {sum(1 for x in mid if x[4])}, kärbitud {sum(1 for x in mid if x[3])}")
print("(L = loop, K = finish_reason length)")
