#!/usr/bin/env python3
"""Kurrendi andmestik v5 (2026-10-12) = v4 (data/kurrent) + ladina + Klinger + DTA <add>.

  - alus data/kurrent (v4 pärast GT-kontrolli väljavõttu, 18 846 rida, holdout 130)
  - DTA (Kosmos, Geusau, Sanders) tekst asendatakse *_v5 ehitusega
    (build_dta_kosmos.py --add, VUTT ADR 0067): rea kohale/alla/veerisele
    kirjutatud lisandus → <add>…</add> lisamiskohas, asenduses kustutatud osa
    välja, pliiatsikäe add/note/fw välja. Ainult asendus: v4-st puuduvaid lehti
    (GT-kontroll võttis välja) juurde ei tule; v5-st puuduv leht kukub välja.
  - Leibniz clean 248 lk (eScriptorium, CC BY 4.0) + Leibniz val: 10 holdout
    (ladina alam-CER, eri dokumentidest, seed) + 17 treeningusse (kasutaja 10.10)
  - Gwalther 142 lk (Zenodo 4780947, CC BY-NC 4.0)
  - Klinger 92 lk (VUTT #576, data/rieger/klapitus/klinger_gt_vutt), holdout 5 = üks
    leht kirja kohta; <m> tavatekstiks (märgendina juhises ainult <add>)
  PAGE XML (Leibniz, Gwalther): regioonid ReadingOrder-i järgi (puudumisel faili
  järjekord), DigitizationArtefactZone ja GraphicZone välja, MarginTextZone ja
  NumberingZone jäävad (nagu teistel PAGE-allikatel); rea tekst rea enda
  TextEquiv-ist. InterlinearLine jääb omaette reaks — lisamiskohta ei ole (ADR 0067).
  Lühendusmärk → makron ja ſ → s kõigis UUTES ridades (nagu v4); Klingeri rea
  lõpu ⸗/= → ¬ (Kurrendi juhis; VUTT-is jääb ⸗).
Pildid hardlink-itud. Sisendkaustu ei muudeta.
Käivitus: venv/bin/python scripts/build_kurrent_v5.py [--dry-run] [--uuesti]
"""
import csv, os, random, re, shutil, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from lyhend_makron import makroniks                                   # noqa: E402

DRY = "--dry-run" in sys.argv
BASE = Path("data/kurrent")
NEW = Path("data/kurrent_v5")
DTA_V5 = [Path("data/dta_kosmos_v5"), Path("data/dta_lisa_v5")]
DTA_ALLIKAD = {"dta_kosmos_1827", "dta_geusau_1740", "dta_sanders_1860"}
RAW = Path("data/raw_xml")
GW = RAW / "gwalther/bullinger-digital-gwalther-handwriting-ground-truth-ef9f482"
KLINGER = Path("data/rieger/klapitus/klinger_gt_vutt")
REGION_VALJA = {"DigitizationArtefactZone", "GraphicZone"}
LEIBNIZ_HO, SEED = 10, 3407
PIKK = 6000                       # tähemärke; ~8192 tokeni piiri lähedal (MAX_SEQ)
LA = set("et in est ad cum non quod qui quae sed ut per ex de ac atque eius esse sunt vel nec enim ab hoc etiam quam autem pro".split())

csv.field_size_limit(sys.maxsize)


def loe(d):
    rows = list(csv.reader(open(d / "metadata.csv", encoding="utf-8")))
    proj = {}
    if (d / "projektid.csv").exists():
        proj = {r["failinimi"]: r["projekt"] for r in csv.DictReader(open(d / "projektid.csv", encoding="utf-8"))}
    return rows[0], rows[1:], proj


def tyyp(el):
    m = re.search(r"structure \{type:(\w+);", el.get("custom", ""))
    return m.group(1) if m else el.get("type", "")


def page_tekst(xml):
    """PAGE XML → lehe tekst; None, kui ridu pole."""
    root = ET.parse(xml).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    page = root.find(f"{ns}Page")
    regs = [r for r in page.iter(f"{ns}TextRegion")]
    ro = [x.get("regionRef") for x in page.iter(f"{ns}RegionRefIndexed")]
    if ro:
        jrk = {rid: i for i, rid in enumerate(ro)}
        regs.sort(key=lambda r: jrk.get(r.get("id"), len(jrk)))
    read = []
    for r in regs:
        if tyyp(r) in REGION_VALJA:
            continue
        for tl in r.findall(f"{ns}TextLine"):
            te = tl.find(f"{ns}TextEquiv")        # rea enda TextEquiv, mitte Word'ide oma
            uc = te.find(f"{ns}Unicode") if te is not None else None
            t = (uc.text or "").strip() if uc is not None else ""
            if t:
                read.append(re.sub(r"\s+", " ", t))
    return "\n".join(read) or None


def ladina(t):
    w = re.findall(r"[a-z]+", t.lower())
    return sum(x in LA for x in w) >= 0.05 * max(len(w), 1)


def page_allikas(xmls, nimi, pilt):
    out = []
    for x in xmls:
        t = page_tekst(x)
        if t:
            out.append((f"images/{nimi}_{x.stem}.jpg", t, nimi, pilt(x), x))
    return out


def main():
    if NEW.exists() and not DRY:
        if "--uuesti" not in sys.argv:
            sys.exit(f"{NEW} on juba olemas (--uuesti kustutab)")
        shutil.rmtree(NEW)
    header, base, proj = loe(BASE)
    ho_read = open(BASE / "holdout.txt", encoding="utf-8").read().splitlines(keepends=True)
    holdout = {os.path.basename(l.strip().split("\t")[-1]) for l in ho_read if l.strip() and not l.startswith("#")}
    st = Counter()

    # 1. DTA tekst v5-st
    dta = {}
    for d in DTA_V5:
        for r in loe(d)[1]:
            dta[r[0]] = r[1]
    out, src, ho_valja, add_n = [], {}, set(), 0
    for r in base:
        if r[2] in DTA_ALLIKAD:
            if r[0] not in dta:
                st["−dta (v5-s puudub)"] += 1
                if os.path.basename(r[0]) in holdout:
                    ho_valja.add(os.path.basename(r[0]))
                continue
            uus = makroniks(dta[r[0]])[0].replace("ſ", "s")
            st["dta tekst muutus" if uus != r[1] else "dta tekst sama"] += 1
            add_n += uus.count("<add>")
            r = [r[0], uus, r[2]]
        out.append(r); src[r[0]] = BASE / r[0]

    # 2. Leibniz, Gwalther, Klinger
    lisa, ho_uued = [], []
    lb_clean = page_allikas(sorted((RAW / "leibniz/train/clean").glob("*.xml")), "leibniz_clean", lambda x: x.with_suffix(".jpg"))
    lb_val = page_allikas(sorted((RAW / "leibniz/val").glob("*.xml")), "leibniz_val", lambda x: x.with_suffix(".jpg"))
    gw = page_allikas(sorted((GW / "page").glob("*.xml")), "gwalther_1540", lambda x: GW / "img" / (x.stem + ".jpeg"))
    rng = random.Random(SEED)
    doc = lambda r: re.sub(r"_page_\d+|\d{4}[rv]?.*$", "", r[4].stem)        # noqa: E731
    kand = [r for r in lb_val if ladina(r[1])]
    rng.shuffle(kand)
    valik, nahtud = [], set()
    for r in kand:                                   # eri dokumentidest esmalt
        if doc(r) not in nahtud and len(valik) < LEIBNIZ_HO:
            valik.append(r); nahtud.add(doc(r))
    for r in kand:
        if len(valik) < LEIBNIZ_HO and r not in valik:
            valik.append(r)
    for r in lb_clean + lb_val + gw:
        lisa.append(r)
        if r in valik:
            ho_uued.append((r[2], r[0]))
    _, kl_rows, _ = loe(KLINGER)
    # Klingeri holdout: üks leht kirja kohta (kasutaja 12.10: 5, treeninguks võimalikult palju)
    kl_kiri = {f"klinger_{r['leht']}.jpg": r["kiri"]
               for r in csv.DictReader(open(KLINGER / "lehed.tsv", encoding="utf-8"), delimiter="\t") if r["holdout"] == "1"}
    kl_ho = {min(f for f, k in kl_kiri.items() if k == kiri) for kiri in set(kl_kiri.values())}
    for r in kl_rows:
        lisa.append((r[0], r[1], r[2], KLINGER / r[0], None))
        if os.path.basename(r[0]) in kl_ho:
            ho_uued.append((r[2], r[0]))
    for f, t, a, pilt, xml in lisa:
        assert f not in src, f
        if not Path(pilt).exists():
            st[f"−pilt puudu ({a})"] += 1
            continue
        t, _ = makroniks(t)
        t = t.replace("ſ", "s")
        if a == "klinger_vutt":   # VUTT-i ⸗/= → Kurrendi juhise ¬ (KURRENT_INSTRUCTION p 3)
            t = re.sub(r"[⸗=]$", "¬", t, flags=re.M)
            t = re.sub(r"</?m>", "", t)   # veeris tavatekstiks (kasutaja 12.10: <m> juhisesse ei lähe)
        out.append([f, t, a]); src[f] = Path(pilt)
        proj[f] = a if xml is None else f"{a}:{xml.stem}"
        st[f"+{a}"] += 1

    pikad = Counter(r[2] for r in out if len(r[1]) > PIKK)
    print(f"v4 {len(base)} → v5 {len(out)}")
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print(f"<add> DTA-s {add_n}, Klingeris {sum(r[1].count('<add>') for r in out if r[2] == 'klinger_vutt')}")
    print(f"holdout −{len(ho_valja)} +{len(ho_uued)}: {Counter(a for a, _ in ho_uued)}")
    print(f"leibniz val holdout: {[os.path.basename(f) for _, f in ho_uued if 'leibniz' in f]}")
    print(f"> {PIKK} tähe (MAX_SEQ oht): {dict(pikad)}")
    if DRY:
        return
    (NEW / "images").mkdir(parents=True)
    for r in out:
        os.link(src[r[0]], NEW / r[0])
    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(out)
    with open(NEW / "projektid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["failinimi", "projekt"])
        w.writerows((r[0], proj.get(r[0], "")) for r in out)
    with open(NEW / "holdout.txt", "w", encoding="utf-8") as f:
        f.writelines(l for l in ho_read
                     if not (l.strip() and not l.startswith("#") and os.path.basename(l.strip().split("\t")[-1]) in ho_valja))
        f.write(f"# v5: −{len(ho_valja)} DTA (v5-s puudub); + Leibniz val {LEIBNIZ_HO} (seed {SEED}) + Klinger {len(kl_ho)}\n")
        f.writelines(f"{a}\t{fn}\n" for a, fn in ho_uued)
    (NEW / "SOURCE.txt").write_text(
        (BASE / "SOURCE.txt").read_text(encoding="utf-8")
        + f"\nehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_kurrent_v5.py\n"
        f"v4 {len(base)} → {len(out)}  {dict(st)}  holdout −{len(ho_valja)} +{len(ho_uued)}\n"
        "DTA <add> (VUTT ADR 0067); Leibniz CC BY 4.0; Gwalther CC BY-NC 4.0; Klinger VUTT #576\n", encoding="utf-8")
    print(f"valmis: {NEW}")


if __name__ == "__main__":
    main()
