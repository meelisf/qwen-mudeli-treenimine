# Miks leht ei saanud <m>-i: otsi v5m rida pildinime järgi, leia XML, näita põhjust
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt
import build_kurrent_v6 as b6
csv.field_size_limit(10 ** 8)
otsi = sys.argv[1]
rows = list(csv.reader(open(b6.BASE / "metadata.csv", encoding="utf-8")))[1:]
r = next(x for x in rows if otsi in x[0])
key = vt.norm(r[1])[:400]
print("v5m rida:", r[0], "| sama võtmega ridu:", sum(1 for x in rows if x[2] == r[2] and vt.norm(x[1])[:400] == key))
for allikas, stem, xs in vt.xmlid():
    if allikas != r[2]:
        continue
    a = vt.analyysi(xs)
    if a and vt.norm(a["tekst"])[:400] == key:
        read, p = b6.uus_tekst(a, r[1].split("\n"))
        print("XML", stem, "→", p, "| klassid:", [(x["klass"], x["read"][0][:20]) for x in a["regioonid"]])
