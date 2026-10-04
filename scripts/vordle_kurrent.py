#!/usr/bin/env python3
"""
Kahe (või enama) Kurrendi eval-jooksu võrdlus holdouti peal.

  python scripts/vordle_kurrent.py kurrent-20260829-Q8_0-h133 kurrent-20261002-Q8_0

Argumendid on kaustanimed `data/kurrent/eval/` all (igaühes `results.csv`,
mille kirjutas `eval_kurrent.py`). Esimene veerg on võrdluse alus.

Mida see teeb ja miks (SEIS §3.4b „Holdouti kontroll"):

- **Kolm pildilekke lehte jäävad välja.** Need olid ka 29.08 treeningus, seega
  mõlemas jooksus ühesugune leke, aga võrdlust nad ainult müraks teevad.
- **Rühmad:** `vanad 70` (29.08 holdout miinus lekked — vana↔uue võrdlus),
  `uued 60` (02.10 lisatud allikad; vana mudel neid treeningus ei näinud, uus
  nägi samade allikate TEISI lehti) ja kõik koos.
- **Otsustav veerg on mediaan loopideta** (ratio ≤ 1,4) — keskmine on
  paksusabaline, üks loop nihutab seda kümneid punkte (vt
  `docs/kurrent-20260829-tulemused.md` §1).
- Võrreldakse ainult lehti, mis on KÕIGIS jooksudes olemas; puudujad
  loetletakse.
"""

import csv
import statistics
import sys
from pathlib import Path

EVAL_ROOT = Path("data/kurrent/eval")
HOLDOUT_PATH = Path("data/kurrent/holdout.txt")
UUTE_MARKER = "# --- Lisatud 2026-10-02"
LOOP_RATIO = 1.4

# Sama pilt on ka treeningus (SEIS §3.4b) — võrdlusest välja
LEKKED = {
    "11771_bullinger_au_1209460_0002_49174186.jpg",
    "16441_aaeb_xiv_xvi_3680035_0013_76516926.jpg",
    "15464_aaeb_xiv_xvi_1627450_0002_60945177.jpg",
}

jooksud = [a for a in sys.argv[1:] if not a.startswith("--")]
if not jooksud:
    print(__doc__)
    sys.exit(1)

# Holdout: vanad = enne markerit, uued = pärast
vanad, uued, allikas_of = set(), set(), {}
on_uus = False
for rida in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines():
    rida = rida.strip()
    if rida.startswith(UUTE_MARKER):
        on_uus = True
        continue
    if not rida or rida.startswith("#"):
        continue
    allikas, failinimi = rida.split("\t")
    nimi = Path(failinimi).name
    if nimi in LEKKED:
        continue
    (uued if on_uus else vanad).add(nimi)
    allikas_of[nimi] = allikas

if len(LEKKED & {Path(r.split("\t")[1]).name for r in
                 HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
                 if r.strip() and not r.startswith("#")}) != 3:
    print("HOIATUS: kõiki 3 lekkelehte holdoutist ei leitud — kas holdout muutus?")

tulemused = {}      # jooks -> {nimi: rida}
for j in jooksud:
    p = EVAL_ROOT / j / "results.csv"
    if not p.exists():
        print(f"Viga: {p} puudub")
        sys.exit(1)
    with open(p, encoding="utf-8", newline="") as f:
        tulemused[j] = {Path(r["failinimi"]).name: r for r in csv.DictReader(f)}

koik = vanad | uued
uhised = set.intersection(*(set(t) & koik for t in tulemused.values()))
for j, t in tulemused.items():
    puudu = koik - set(t)
    if puudu:
        print(f"NB! {j}: {len(puudu)} holdout-lehte puudub (nt {sorted(puudu)[0]})")


def cer(j, n):
    return float(tulemused[j][n]["cer"])


def loop(j, n):
    return float(tulemused[j][n]["ratio"]) > LOOP_RATIO


def pct(x):
    return "—" if x is None else f"{x:.1%}".replace(".", ",")


def med(xs):
    return statistics.median(xs) if xs else None


def kesk(xs):
    return statistics.fmean(xs) if xs else None


print(f"\nAlus: {jooksud[0]}.  Loop = ratio > {LOOP_RATIO}.  Lekked (3) väljas.")

for ryhm, nimed in [("vanad 70", vanad), ("uued 60", uued), ("kõik", koik)]:
    nimed = sorted(nimed & uhised)
    print(f"\n## {ryhm} ({len(nimed)} lk)\n")
    print("| jooks | keskm | med | loope | loopideta keskm | **med** |")
    print("|---|---|---|---|---|---|")
    for j in jooksud:
        c = [cer(j, n) for n in nimed]
        puhas = [cer(j, n) for n in nimed if not loop(j, n)]
        loope = sum(loop(j, n) for n in nimed)
        print(f"| {j} | {pct(kesk(c))} | {pct(med(c))} | {loope} "
              f"| {pct(kesk(puhas))} | **{pct(med(puhas))}** |")

# Paaris: lehed, kus kumbki jooks loopis, jäävad välja (muidu domineerivad)
alus = jooksud[0]
for j in jooksud[1:]:
    nimed = [n for n in uhised if not loop(alus, n) and not loop(j, n)]
    vahed = [cer(j, n) - cer(alus, n) for n in nimed]
    parem = sum(v < -0.01 for v in vahed)
    halvem = sum(v > 0.01 for v in vahed)
    print(f"\n{j} vs {alus}: {len(nimed)} loopita lehte — "
          f"parem {parem}, halvem {halvem}, ±1 pp sees {len(nimed) - parem - halvem}; "
          f"mediaanvahe {pct(med(vahed))}")

# Allika kaupa, mediaan loopideta
print("\n## Allika kaupa (mediaan-CER, loopideta)\n")
print("| allikas | rühm | lk | " + " | ".join(jooksud) + " |")
print("|---|---|---|" + "---|" * len(jooksud))
for allikas in sorted({allikas_of[n] for n in uhised}):
    nimed = [n for n in uhised if allikas_of[n] == allikas]
    ryhm = "uus" if nimed[0] in uued else "vana"
    veerud = [pct(med([cer(j, n) for n in nimed if not loop(j, n)])) for j in jooksud]
    print(f"| {allikas} | {ryhm} | {len(nimed)} | " + " | ".join(veerud) + " |")

# Loopid nimeliselt — need tuleb silmaga üle vaadata
print("\n## Loopid ja lühikesed väljundid\n")
for j in jooksud:
    pikad = sorted(n for n in uhised if loop(j, n))
    luhikesed = sorted(n for n in uhised if float(tulemused[j][n]["ratio"]) < 0.7)
    print(f"{j}: loop {pikad or '—'}; lühike (<0,7) {luhikesed or '—'}")
