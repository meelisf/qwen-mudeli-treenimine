#!/usr/bin/env python3
"""Kurrendi andmestik v4 (2026-10-04): iga allikas auditeeritud.

  - alus data/kurrent_v3 (v2 miinus ≥2 tühja TextLine-iga xix-lehed)
  - bullinger_autoren asendatakse data/bullinger_v3-ga (parim XML-versioon,
    ≤1 tühi rida); holdout-lehed jäävad, et eval oleks võrreldav
  - xix-i parthey/hufeland/nn_msgermqu (Matcheri osaline GT) asendatakse
    DTA täistekstiga: data/dta_kosmos (build_dta_kosmos.py), + 5 uut kätt
  - Escheri kõik puhtad lehed: data/escher_lisa
  - dresdner_1665 asendatakse data/dresdner_v2-ga (õige pb piir, tabeliga
    lehed välja); holdout-lehe tekst võetakse v2-st, tabelileht kukub välja
  - lühendusmärk tilde/ülakriips → makron kõigis ridades (VUTT ADR 0062)
  - holdout: + Geusau 10 ja Kosmos 10 (iga käsikiri korra), vt lisa_holdout
  - AUDIT: iga AUDITID-faili treeningrida, millel ≥ LAVI tühja TextLine'i,
    läheb välja — ka holdout'ist (katkine GT ei sobi ka mõõtmiseks)
Pildid hardlink-itud. Sisendkaustu ei muudeta.
Käivitus: venv/bin/python scripts/build_kurrent_v4.py [--dry-run] [--uuesti]
  --uuesti: olemasolev data/kurrent_v4 kustutatakse (ainult lingid + CSV)
"""
import csv, os, random, re, shutil, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lyhend_makron import makroniks                                   # noqa: E402

DRY = "--dry-run" in sys.argv
BASE = Path("data/kurrent_v3")
BULL = Path("data/bullinger_v3")
DTA = Path("data/dta_kosmos")
ESCH = Path("data/escher_lisa")
DRES = Path("data/dresdner_v2")
DTA2 = Path("data/dta_lisa")
NEW = Path("data/kurrent_v4")
DTA_XIX = {"parthey", "hufeland_privatbesitz_1829", "nn_msgermqu2124_1827", "nn_msgermqu2345_1827"}
LAVI = 2
HO_SEED, HO_MIN_CHARS = 3407, 200               # = make_holdout.py
HO_GEUSAU, HO_KOSMOS = 10, 10
A = Path("data/kurrent_xix_audit")
AUDITID = [A / "tuhjad_read_kaug.csv",          # AAEB, Königsfelden (Bullinger: holdout)
           A / "tuhjad_read_riksarkivet.csv",   # 7 Riksarkiveti allikat
           A / "tuhjad_read_senats.csv",
           A / "tuhjad_read_zurich.csv"]

csv.field_size_limit(sys.maxsize)


def loe(d):
    rows = list(csv.reader(open(d / "metadata.csv", encoding="utf-8")))
    proj = {r["failinimi"]: r["projekt"] for r in csv.DictReader(open(d / "projektid.csv", encoding="utf-8"))}
    return rows[0], rows[1:], proj


def katkised(gt_ridu):
    """Treeningu failinimi (basename) → tühjade arv, kui ≥ LAVI.
    Mitme XML-versiooni korral valitakse see, mille täidetud ridade arv
    klapib GT ridade arvuga (Bullinger)."""
    ver = defaultdict(list)
    for f in AUDITID:
        if not f.exists():
            if DRY:
                print(f"HOIATUS: {f} puudub (kuivkäivitus jätkab)")
                continue
            sys.exit(f"{f} puudub — kõik allikad peavad olema auditeeritud")
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["treeningus"]:
                ver[os.path.basename(r["treeningus"])].append((int(r["tuhje"]), int(r["ridu"])))
    out = {}
    for name, V in ver.items():
        n = gt_ridu.get(name)
        sobiv = [v for v in V if n is not None and v[1] - v[0] == n] or V
        e = min(v[0] for v in sobiv)
        if e >= LAVI:
            out[name] = e
    return out


def lisa_holdout(out, proj):
    """Uued DTA allikad holdout'i (kasutaja 04.10): muidu ei mõõdeta just neid
    käsi, mis on VUTT-ile lähimad. Geusau HO_GEUSAU lk; Kosmos HO_KOSMOS lk, iga
    käsikiri vähemalt korra. Seed ja min_chars nagu make_holdout.py-s."""
    rng = random.Random(HO_SEED)
    sobib = [r for r in out if len(r[1]) >= HO_MIN_CHARS]
    geusau = sorted(r[0] for r in sobib if r[2] == "dta_geusau_1740")
    valik = rng.sample(geusau, HO_GEUSAU)
    kosmos = defaultdict(list)
    for r in sobib:
        if r[2] == "dta_kosmos_1827":
            kosmos[proj[r[0]]].append(r[0])
    kasikirjad = sorted(kosmos)
    for k in kasikirjad:                       # iga käsi korra
        valik.append(rng.choice(sorted(kosmos[k])))
    jaak = sorted(f for k in kasikirjad for f in kosmos[k] if f not in valik)
    valik += rng.sample(jaak, HO_KOSMOS - len(kasikirjad))
    allikas = {r[0]: r[2] for r in out}
    return [(allikas[f], f) for f in valik]


def main():
    if NEW.exists() and not DRY:
        if "--uuesti" not in sys.argv:
            sys.exit(f"{NEW} on juba olemas (--uuesti kustutab)")
        shutil.rmtree(NEW)
    header, base, proj = loe(BASE)
    ho_read = open(BASE / "holdout.txt", encoding="utf-8").read().splitlines(keepends=True)
    holdout = {os.path.basename(l.strip().split("\t")[-1]) for l in ho_read if l.strip() and not l.startswith("#")}
    gt_ridu = {os.path.basename(r[0]): sum(1 for l in r[1].split("\n") if l.strip()) for r in base}
    halvad = katkised(gt_ridu)
    _, dres_rows, _ = loe(DRES)
    dres = {int(re.search(r"(\d+)\.jpg$", r[0]).group(1)): r for r in dres_rows}

    out, src, st, ho_valja, dres_ho = [], {}, Counter(), set(), set()
    for r in base:
        b = os.path.basename(r[0])
        ho = b in holdout
        if b in halvad:
            st[f"−audit ≥{LAVI} tühja ({r[2]})"] += 1
            if ho:
                ho_valja.add(b)
            continue
        if r[2] == "dresdner_1665":
            nr = int(re.search(r"(\d+)\.jpg$", r[0]).group(1))
            if not ho:
                st["−dresdner vana"] += 1
                continue
            if nr not in dres:                     # tabelileht v2-s välja
                st["−dresdner holdout (tabel)"] += 1
                ho_valja.add(b)
                continue
            r = [r[0], dres[nr][1], r[2]]          # holdout: sama pilt, parandatud tekst
            dres_ho.add(nr)
        if r[2] == "bullinger_autoren" and not ho:
            st["−bullinger vana"] += 1
            continue
        if proj.get(r[0]) in DTA_XIX and not ho:
            st["−xix matcher (DTA asemel)"] += 1
            continue
        out.append(r); src[r[0]] = BASE
    for d, nimi in ((BULL, "+bullinger v3"), (DTA, "+dta kosmos"), (ESCH, "+escher puhtad"), (DRES, "+dresdner v2"), (DTA2, "+dta lisa")):
        _, rows, p = loe(d)
        for r in rows:
            if d == DRES and int(re.search(r"(\d+)\.jpg$", r[0]).group(1)) in dres_ho:
                continue
            assert r[0] not in src, r[0]
            out.append(r); src[r[0]] = d; proj[r[0]] = p[r[0]]
            st[nimi] += 1
    # Lühendusmärk → makron kõigis ridades, sh holdout (ADR 0062). makroniks
    # tagastab NFC — rakendatakse KÕIGILE, sest lahutatud „e + U+0304" ja „ē" on
    # mudelile kaks eri järjestust (Königsfelden, Hanse, Senats).
    # Pikk s → s kõigis ridades (06.10): ſ oli ainult senats/dresdner/osa xix-ist
    # (445/18 846 lehte), juhis käskis ſ-i — mudel nägi vastuolu. Kurrendis on
    # pikk s niikuinii reegel, VUTT-i toimetajad seda ei erista.
    mk, nfc = Counter(), 0
    for i, r in enumerate(out):
        t, n = makroniks(r[1])
        t = t.replace("ſ", "s")
        mk[r[2]] += n
        if t != r[1]:
            nfc += not n
            out[i] = [r[0], t, r[2]]
    mk = Counter({k: v for k, v in mk.items() if v})
    ho_uued = lisa_holdout(out, proj)
    print(f"v3 {len(base)} → v4 {len(out)}")
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print(f"holdout −{len(ho_valja)}: {sorted(ho_valja)}")
    print(f"makron (märke allika kaupa): {dict(mk.most_common())}; ainult NFC-ga muutunud ridu {nfc}")
    print(f"holdout +{len(ho_uued)} DTA: {Counter(proj[f] for _, f in ho_uued)}")
    print(Counter(r[2] for r in out).most_common())
    if DRY:
        return
    (NEW / "images").mkdir(parents=True)
    for r in out:
        os.link(src[r[0]] / r[0], NEW / r[0])
    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(out)
    with open(NEW / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "projekt"])
        w.writerows((r[0], proj.get(r[0], "")) for r in out)
    with open(NEW / "holdout.txt", "w", encoding="utf-8") as f:
        if ho_valja:
            f.write(f"# v4: eemaldatud {len(ho_valja)} lehte (auditi ≥{LAVI} tühja rida / Dresdneri tabel)\n")
        f.writelines(l for l in ho_read
                     if not (l.strip() and not l.startswith("#") and os.path.basename(l.strip().split("\t")[-1]) in ho_valja))
        f.write(f"# v4: lisatud {len(ho_uued)} DTA lehte (Geusau {HO_GEUSAU}, Kosmos {HO_KOSMOS}; seed {HO_SEED})\n")
        f.writelines(f"{a}\t{fn}\n" for a, fn in ho_uued)
    (NEW / "SOURCE.txt").write_text(
        (BASE / "SOURCE.txt").read_text(encoding="utf-8")
        + f"\nehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_kurrent_v4.py\n"
        f"v3 {len(base)} → {len(out)}  {dict(st)}  holdout −{len(ho_valja)} +{len(ho_uued)}\n"
        f"lühendusmärk → makron (ADR 0062, scripts/lyhend_makron.py): {sum(mk.values())} märki\n", encoding="utf-8")
    print(f"valmis: {NEW}")


if __name__ == "__main__":
    main()
