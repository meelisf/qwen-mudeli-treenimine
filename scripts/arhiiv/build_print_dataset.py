#!/usr/bin/env python3
"""
Trükitreeningu andmestik: `data/lehekyljed` + Gezeliuse parandused

Baasilt treenimine (plaani samm 9) nõuab, et 1. etapi 1500 lehte oleksid
treeningkomplektis otseselt — muidu kaob kreeka signaal (559 kreekarikkast
lehest 549 on Gezelius). Aga `data/lehekyljed/metadata.csv` on 11.11.2025
loodud ja muutumatu, ning Gezeliuse kaks teost on seal **moderniseeritud**:

    ſ leidub lehtedel:  Lexicon 0/447,  Ianua 0/273,  ülejäänu 722/780
    (VUTT-i korpuses 1105/1113)

Ehk just need 720 lehte on ainsad, mis õpetaksid mudelile vastupidist seda,
mida ülejäänud 2 340 lehte õpetavad. See skript teeb neist parandatud koopia:

  Lexicon_exact (447)  homoglüüfid + ligatuurid + pikk ſ + pseudomärgendus `<i>`
  Comenius-Ianua (273) homoglüüfid + ligatuurid + pikk ſ (märgendust ei ole)
  ülejäänud (780)      muutmata

Ianua kontrollitud originaalpildilt (lk 0033: „Mare ſalſum eſt, muriæ inſtar.")
— sama trükikonventsioon mis Lexiconis, seega sama parandus. Pseudomärgendust
Ianua EI saa: ta on paralleelküljendus (0,2 % segaridu), kus „kreeka märksõna +
kursiivne ladina gloss" struktuuri ei ole.

Väljund läheb eraldi faili, originaali ei puudutata.

Käivitamine:
  python scripts/build_print_dataset.py
  python scripts/build_print_dataset.py --out data/lehekyljed/muu.csv
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gezelius_lexicon import (  # noqa: E402
    _is_running_head,
    fix_homoglyphs,
    fix_ligatures,
    fix_long_s,
    fix_running_head,
)

csv.field_size_limit(10 ** 7)

SISEND = Path("data/lehekyljed/metadata.csv")
VÄLJUND = Path("data/lehekyljed/metadata_markup.csv")
PSEUDO = Path("data/export/gezelius-lexicon-pseudo")

LEXICON = re.compile(r"(\d+)_Gezelius_Lexicon_exact")
IANUA = re.compile(r"(\d+)_Gezelius-Comenius-Ianua")


def paranda(tekst: str) -> str:
    """Homoglüüfid + ligatuurid + pikk ſ, päisemarkerid eraldi."""
    tekst = fix_long_s(fix_ligatures(fix_homoglyphs(tekst)))
    read = tekst.split("\n")
    if read and _is_running_head(read[0]):
        read[0] = fix_running_head(read[0])
        tekst = "\n".join(read)
    return tekst


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else VÄLJUND

    if not SISEND.exists():
        sys.exit(f"Viga: {SISEND} puudub")
    if not PSEUDO.exists():
        sys.exit(f"Viga: {PSEUDO} puudub – jooksuta enne "
                 f"`python scripts/gezelius_lexicon.py --pseudo {PSEUDO}`")

    read = list(csv.DictReader(open(SISEND, encoding="utf-8")))
    n_lex = n_ianua = n_puudu = 0

    for r in read:
        nimi = r["failinimi"].split("/")[-1]
        m = LEXICON.match(nimi)
        if m:
            fail = PSEUDO / f"gezelius-lexicon-{int(m.group(1)):04d}.txt"
            if not fail.exists():
                n_puudu += 1
                continue
            r["transkriptsioon"] = fail.read_text(encoding="utf-8").rstrip()
            n_lex += 1
            continue
        if IANUA.match(nimi):
            r["transkriptsioon"] = paranda(r["transkriptsioon"])
            n_ianua += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["failinimi", "transkriptsioon"])
        w.writeheader()
        w.writerows({"failinimi": r["failinimi"],
                     "transkriptsioon": r["transkriptsioon"]} for r in read)

    kogu = "".join(r["transkriptsioon"] for r in read)
    print(f"Kirjutatud {len(read)} rida → {out}")
    print(f"  Lexicon pseudomärgendusega: {n_lex}")
    print(f"  Ianua parandatud:           {n_ianua}")
    if n_puudu:
        print(f"  HOIATUS: pseudofail puudus {n_puudu} Lexiconi lehel")
    print(f"  ſ kokku: {kogu.count('ſ')}, æ/œ: {kogu.count('æ') + kogu.count('œ')}, "
          f"<i>: {kogu.count('<i>')}")


if __name__ == "__main__":
    main()
