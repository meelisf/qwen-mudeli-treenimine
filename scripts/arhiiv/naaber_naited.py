# naaberleht-regioonide näited allika kaupa (tekst + laius/kaugus), et kontrollida laiuse reeglit
import sys, random
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
allikas_soov = sys.argv[1] if len(sys.argv) > 1 else "svea_hovratt_seg"
n = defaultdict(list)
for allikas, stem, xs in vt.xmlid():
    if allikas != allikas_soov:
        continue
    a = vt.analyysi(xs)
    if not a:
        continue
    pw = max(r["bbox"][2] - r["bbox"][0] for r in a["regioonid"] if r["klass"] == "põhi")
    for r in a["regioonid"]:
        if r["klass"] == "naaberleht":
            n["k"].append((stem[-22:], round((r["bbox"][2] - r["bbox"][0]) / pw, 2), r.get("kaugus"), len(r["read"]), " / ".join(r["read"][:2])[:60]))
random.seed(3)
for x in random.sample(n["k"], min(15, len(n["k"]))):
    print(x)
