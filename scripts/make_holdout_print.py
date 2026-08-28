#!/usr/bin/env python3
"""
Trükikorpuse (VUTT) holdout-nimekiri

Plaani samm 8. Ilma holdoutita ei ole uut trükimudelit millegagi ausalt
mõõta: kõik VUTT-i lehed on treeningus ja „vaatame väljundeid" jääb
muljepõhiseks.

Valik on **marginaalikeskne**, sest `<m>` on ainus sine qua non
(docs/arhiiv/plaan-trukipool-jargmine-treening.md, 0b). Kolm kihti:

  marginaalirohke  10 lk  – kus `<m>` on ja kus mudel tavaliselt eksib
  marginaaliga      5 lk  – 1–3 `<m>`, tavaline juht
  ilma              5 lk  – kontroll: kas mudel hakkab marginaale luuletama

Valik on deterministlik (seed) ja läheb repositooriumi, et sama komplekt
kehtiks ka järgmise mudeli juures.

Käivitamine:
  python scripts/make_holdout_print.py --stats   # ainult näita
  python scripts/make_holdout_print.py           # kirjutab data/vutt/holdout.txt
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from convert_marginalia import clean_markup  # noqa: E402

csv.field_size_limit(10 ** 7)

DATA_CSV = Path("data/vutt/metadata.csv")
OUT_PATH = Path("data/vutt/holdout.txt")

SEED = 3407
KIHID = [("marginaalirohke", 10), ("marginaaliga", 5), ("ilma", 5)]
MIN_CHARS = 200      # liiga lühike leht ei anna alust
ROHKE_PIIR = 4       # `<m>` tage, millest alates leht loeb marginaalirohkeks

DRY_RUN = "--stats" in sys.argv
for i, a in enumerate(sys.argv):
    if a == "--seed" and i + 1 < len(sys.argv):
        SEED = int(sys.argv[i + 1])
    elif a == "--out" and i + 1 < len(sys.argv):
        OUT_PATH = Path(sys.argv[i + 1])


def kiht(n_m: int) -> str:
    if n_m >= ROHKE_PIIR:
        return "marginaalirohke"
    return "marginaaliga" if n_m else "ilma"


def main() -> None:
    if not DATA_CSV.exists():
        sys.exit(f"Viga: {DATA_CSV} puudub")

    kaupa: dict[str, list] = {k: [] for k, _ in KIHID}
    for r in csv.DictReader(open(DATA_CSV, encoding="utf-8")):
        t = clean_markup(r.get("transkriptsioon", ""))
        if len(t) < MIN_CHARS:
            continue
        n_m = t.count("<m>")
        kaupa[kiht(n_m)].append((Path(r["failinimi"]).name, n_m, len(t)))

    rng = random.Random(SEED)
    valik = []
    for nimi, mitu in KIHID:
        olemas = sorted(kaupa[nimi])
        print(f"{nimi:16s} kandidaate {len(olemas):4d}, valin {mitu}")
        if len(olemas) < mitu:
            print(f"  HOIATUS: kandidaate on vähem kui vaja")
        valik += rng.sample(olemas, min(mitu, len(olemas)))

    valik.sort()
    print(f"\nHoldout: {len(valik)} lehte, "
          f"{sum(m for _, m, _ in valik)} <m> tagi kokku")
    for nimi, n_m, pikkus in valik:
        print(f"  {nimi:55s} <m>={n_m:3d}  {pikkus:6d} tähemärki")

    if DRY_RUN:
        print("\n--stats: ei kirjutanud midagi.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        f"# VUTT trükikorpuse holdout – scripts/make_holdout_print.py, seed {SEED}\n"
        f"# {len(valik)} lehte: 10 marginaalirohket, 5 marginaaliga, 5 ilma.\n"
        f"# Need lehed EI OLE treeningus – nende peal võrreldakse mudeleid.\n"
        + "".join(f"{n}\n" for n, _, _ in valik),
        encoding="utf-8",
    )
    print(f"\nKirjutatud → {OUT_PATH}")


if __name__ == "__main__":
    main()
