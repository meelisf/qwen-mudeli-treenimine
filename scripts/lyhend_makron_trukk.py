#!/usr/bin/env python3
"""Trükimudeli treeningandmed: lühendusmärk → makron + prügi (VUTT ADR 0062, #533 samm 4).

Failid (git'is): data/lehekyljed/metadata.csv, data/lehekyljed/metadata_markup.csv,
data/vutt/metadata.csv. Kuivkäivitus vaikimisi; `--kirjuta` kirjutab. CSV vorming
(QUOTE_MINIMAL, faili oma reavahetus) säilib — muutmata rida on baithaaval sama.

Teisendus rea kaupa, järjekorras:
  1. U+E8BF (MUFI q-ligatuur „que") → „q;" — andmestiku valitsev kuju (q; ~6 000 korda);
     ligatuurile järgnev kombineeriv märk on prügi ja kukub koos temaga
  2. U+F1A7 → „I" (kursiivne suur I, 266_PDFsam_…_page_3: „Inde & Philoſophos", pildilt)
  3. kombineeriv märk REA ALGUSES (alus puudub) → maha
  4. kreeka täht + U+0303 → U+0342 (perispomeni; lyhend_makron jätab kreeka puutumata)
  5. makroniks (lyhend_makron.py: ladina täht + tilde/ülakriips → makron, NFC)

Keelevalvurit (est/spa/por) ei ole vaja: mõõdetud 2026-10-05 — data/vutt 962/1120 lehe
teosel `languages` valvurita, ülejäänud 158 on kärbitud kaustanimega 1626–1802 ladina/saksa
teosed; data/lehekyljed `õ`-d on kõik lühendid (#533). Mudeli väljundeid (data/vutt/eval,
reocr) EI muudeta — eval võrdsustab tilde ja makroni (textmetrics, eval_print).
"""
import argparse
import csv
import io
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lyhend_makron import makroniks

csv.field_size_limit(10 ** 8)

FAILID = ["data/lehekyljed/metadata.csv", "data/lehekyljed/metadata_markup.csv",
          "data/vutt/metadata.csv"]

QUE = re.compile("\ue8bf[\u0300-\u036f]*")
REA_ALGUS = re.compile(r"(?m)^[\u0300-\u036f]+")


def _kreeka_tilde(s, c):
    d = ud.normalize("NFD", s)
    out, alus = [], ""
    for ch in d:
        if ud.combining(ch) == 0:
            alus = ch
        elif ch == "\u0303" and "GREEK" in ud.name(alus, ""):
            ch = "\u0342"
            c["kreeka tilde → perispomeni"] += 1
        out.append(ch)
    return ud.normalize("NFC", "".join(out))


def puhasta(t, c):
    t, n = QUE.subn("q;", t);              c["U+E8BF → q;"] += n
    n = t.count("\uf1a7"); t = t.replace("\uf1a7", "I"); c["U+F1A7 → I"] += n
    t, n = REA_ALGUS.subn("", ud.normalize("NFD", t)); c["kombineeriv rea alguses"] += n
    t = _kreeka_tilde(t, c)
    t, n = makroniks(t);                   c["tilde/ülakriips → makron"] += n
    return t


def _test():
    c = Counter()
    juhud = [
        ("iudiciumq; timent abs\ue8bf d", "iudiciumq; timent absq; d"),
        ("Canterur\ue8bf fera", "Canterurq; fera"),
        ("r\ue8bf\u0303 O", "rq; O"),
        ("\uf1a7nde & Philoſophos", "Inde & Philoſophos"),
        ("Pythagora\n\u0303εσω σίνδονος", "Pythagora\nεσω σίνδονος"),
        ("x\n\u0301ἄλυτος", "x\nἄλυτος"),
        ("didυ\u0303oV", "didῦoV"),
        ("cũ nõ Camm\u0303erherr", "cū nō Camm\u0304erherr"),
        ("ꝗ cy ſont", "ꝗ cy ſont"),          # MUFI-standardne ꝗ jääb
    ]
    for sisend, oodatud in juhud:
        tul = puhasta(sisend, c)
        assert tul == ud.normalize("NFC", oodatud), (sisend, tul, oodatud)
    print(f"OK: {len(juhud)} juhtu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kirjuta", action="store_true", help="kirjuta failid (muidu kuivkäivitus)")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        return _test()
    for p in FAILID:
        raw = Path(p).read_text(encoding="utf-8")
        lt = "\r\n" if "\r\n" in raw.split("\n", 1)[0] + "\n" else "\n"
        rows = list(csv.reader(io.StringIO(raw, newline="")))
        c, lehti = Counter(), 0
        for r in rows[1:]:
            uus = puhasta(r[1], c)
            if uus != r[1]:
                lehti += 1
                r[1] = uus
        jaak = sum(ud.normalize("NFD", r[1]).count("\u0303") for r in rows[1:])
        print(f"== {p}: {len(rows) - 1} lehte, muutus {lehti}; tilde jääk {jaak}")
        for k, v in c.items():
            print(f"   {k}: {v}")
        if a.kirjuta:
            b = io.StringIO()
            csv.writer(b, lineterminator=lt).writerows(rows)
            Path(p).write_text(b.getvalue(), encoding="utf-8", newline="")
    if not a.kirjuta:
        print("\nKuivkäivitus. Kirjutamiseks: --kirjuta")


if __name__ == "__main__":
    main()
