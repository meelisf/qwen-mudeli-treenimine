# Märgijada kattuvus: kui suur osa veerise tekstist (normaliseeritud) leidub põhitekstis järjestikuste plokkidena
import difflib, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
def n(t): return re.sub(r"[^a-zäöüß]", "", t.lower().replace("ſ", "s"))
def kattuvus(m, p):
    sm = difflib.SequenceMatcher(None, m, p, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks() if b.size >= 6) / max(len(m), 1)
SOOV = set(sys.argv[1:]); tul = []
for allikas, stem, xs in vt.xmlid():
    if allikas not in SOOV:
        continue
    a = vt.analyysi(xs)
    if not a:
        continue
    v = [t for r in a["regioonid"] if r["klass"].startswith("veeris") for t, _ in r["ridade_bbox"]]
    m = n("".join(v)); p = n("".join(t for r in a["regioonid"] if r["klass"] in ("põhi", "teine_plokk") for t, _ in r["ridade_bbox"]))
    if len(m) >= 150:
        tul.append((round(len(m) / max(len(p), 1), 2), allikas[:16], stem[-24:], len(v), " / ".join(v[:3])[:55]))
tul.sort(reverse=True)
for x in tul[:14]: print(x)
print("...")
import collections
print("jaotus:", collections.Counter(round(x[0], 1) for x in tul))
