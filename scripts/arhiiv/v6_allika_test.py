# Ühe allika kiirtest: XML ↔ v5m sidumine, uus_tekst põhjused, veerise ridu
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
import build_kurrent_v6 as b6
csv.field_size_limit(10 ** 8)
soov = set(sys.argv[1:])
rows = list(csv.reader(open(b6.BASE / "metadata.csv", encoding="utf-8")))[1:]
idx = defaultdict(lambda: defaultdict(list))
for k, r in enumerate(rows):
    if r[2] in soov:
        idx[r[2]][vt.norm(r[1])[:400]].append(k)
c = defaultdict(Counter); naide = {}
for allikas, stem, xs in vt.xmlid():
    if allikas not in soov:
        continue
    a = vt.analyysi(xs)
    if not a:
        c[allikas]["parse_viga"] += 1; continue
    ks = idx[allikas].get(vt.norm(a["tekst"])[:400], [])
    c[allikas]["seotud" if ks else "sidumata"] += 1
    for k in ks[:1]:
        read, p = b6.uus_tekst(a, rows[k][1].split("\n"))
        c[allikas][p] += 1
        if p == "veeris_m" and allikas not in naide:
            naide[allikas] = (rows[k][0], "\n".join(read)[:700])
for al in soov:
    print(al, "v5m ridu", sum(len(v) for v in idx[al].values()), dict(c[al]))
for al, (f, t) in naide.items():
    print("\n---", f); print(t)
