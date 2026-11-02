# Rööpveeru kandidaadid: pikk veerise regioon — kas tekst kordab põhiteksti (rööptekst) või on päris veeris?
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
SOOV = set(sys.argv[1:])
def sonad(t): return set(w.lower() for w in re.findall(r"\w{4,}", t))
for allikas, stem, xs in vt.xmlid():
    if allikas not in SOOV:
        continue
    a = vt.analyysi(xs)
    if not a:
        continue
    pohi = " ".join(t for r in a["regioonid"] if r["klass"] in ("põhi", "teine_plokk") for t, _ in r["ridade_bbox"])
    pw = max(r["bbox"][2] - r["bbox"][0] for r in a["regioonid"] if r["klass"] == "põhi")
    for r in a["regioonid"]:
        if r["klass"].startswith("veeris") and len(r["ridade_bbox"]) >= 12:
            m = " ".join(t for t, _ in r["ridade_bbox"])
            s = sonad(m); kordus = len(s & sonad(pohi)) / max(len(s), 1)
            print(f"{allikas[:18]:18s} {stem[-22:]:22s} ridu {len(r['ridade_bbox']):3d} laius {round((r['bbox'][2]-r['bbox'][0])/pw,2)} kordus {kordus:.2f}  {m[:60]!r}")
