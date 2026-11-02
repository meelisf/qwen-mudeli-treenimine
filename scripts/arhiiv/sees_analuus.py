# Ploki-sisesed veerisekandidaadid: kumb pool, üksik vs rühm, number vs tekst — sümmeetrilise reegli jaoks
import re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
c = Counter(); lehed = Counter(); naited = {}
for allikas, stem, xs in vt.xmlid():
    a = vt.analyysi(xs)
    if not a:
        continue
    kahtlane = False
    for r in a["regioonid"]:
        sees = r.get("veeris_sees", [])
        if not sees:
            continue
        rb = [(t, bb) for t, bb in r["ridade_bbox"] if bb]
        jarjestus = {t: i for i, (t, _) in enumerate(sorted(rb, key=lambda x: x[1][1]))}
        kand = sorted(((jarjestus.get(t, -9), t, k) for t, k in sees))
        for j, (i, t, k) in enumerate(kand):
            naaber = any(abs(i - i2) == 1 and k2 == k for i2, _, k2 in kand)
            num = re.fullmatch(r"[\d\s.,:/§]{1,8}", t) is not None
            liik = "number" if num else "rühm" if naaber else ("üksik_lühike" if len(t) <= 14 else "üksik_pikk")
            c[(k, liik)] += 1
            if liik.startswith("üksik"):
                kahtlane = True
            naited.setdefault((k, liik), []).append(t)
    if kahtlane:
        lehed[allikas] += 1
for (k, l), n in sorted(c.items()):
    print(f"{k:6s} {l:13s} {n:4d}  nt {naited[(k, l)][:6]}")
print("lehti, kus on üksik mittenumbriline kandidaat:", dict(lehed), sum(lehed.values()))
