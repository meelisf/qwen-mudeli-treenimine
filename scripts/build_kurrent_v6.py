#!/usr/bin/env python3
"""Kurrendi andmestik v6 (2026-10-14) = v5m + veeris <m>-ina PAGE XML-i allikates.

v5-s kandis <m>-i ainult Leibniz + DTA; PAGE XML-i allikates oli veeris tavaridadena
lugemisjärjekorras (sageli lehe alguses/lõpus) → mudel pani kantseleikäte veerise
märgendita lõppu (VUTT agnp5i, 14.10). v6 teeb Riksarkiveti 7 allikas ja Tübingeni
senatsprotokollides veerise `<m>…</m>`-ridadeks omal kohal (iga füüsiline rida
omaette, VUTT ADR 0009; paigutus nagu v5 Leibnizil: peremeesploki ette esimese
põhirea ette, mille keskpunkt on temast allpool).

Tuvastus: scripts/veeris_tuvastus.py (geomeetria; reeglid kasutaja kontrolliga 14.10).
Reeglid (SÜMMEETRILISED — kasutaja 14.10: ühe poole eelistamine õpetaks teise poole
veerise vahelejätmist):
  - veeris eraldi regioonina → <m>, mõlemal pool;
  - veeris põhiploki SEES → <m> ainult kui rida on number („11.", „5");
  - muu ploki-sisene kandidaat (taandrida, ületäie rea sõna, tabelirida või päris
    veeris — ei erista) → KOGU LEHT VÄLJA, mõlemal pool;
  - kahe lehekülje laotus ühes regioonis → leht välja;
  - naaberlehe/dorsaali tekst → read GT-st välja (pildil näha, aga ei kuulu lehele;
    sama reegel mis VUTT Gemini juhises, PR #600).
Tekst võetakse v5m GT-st REAINDEKSI järgi (makron, ſ→s jm v4/v5 teisendused jäävad);
kui XML-i ja GT ridade arv ei klapi ja lehel on midagi muuta → leht välja.

Pildid hardlink-itud. Sisendkaustu ei muudeta. Holdout = v5m holdout miinus väljajäetud.

  venv/bin/python scripts/build_kurrent_v6.py [--dry-run]
"""
import csv, os, re, shutil, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import veeris_tuvastus as vt                                             # noqa: E402

csv.field_size_limit(10 ** 8)
BASE = Path("data/kurrent_v5m")
NEW = Path("data/kurrent_v6")
DRY = "--dry-run" in sys.argv
NUMBER = re.compile(r"[\d\s.,:/§]{1,8}")


def yk(bb):
    return (bb[1] + bb[3]) / 2 if bb else None


def uus_tekst(a, gt_read):
    """(uued read | None, põhjus). a = vt.analyysi tulemus, gt_read = v5m GT read."""
    regs = a["regioonid"]
    xml_read = [(r, i) for r in regs for i in range(len(r["ridade_bbox"]))]
    if a["laotus_yhes"]:
        return None, "laotus_yhes"
    plokid = [r for r in regs if r["klass"] in ("põhi", "teine_plokk")]
    sees_num, sees_muu = set(), False
    for blk in plokid:
        for i in blk.get("veeris_sees_idx", []):
            if NUMBER.fullmatch(blk["ridade_bbox"][i][0]):
                sees_num.add((id(blk), i))
            else:
                sees_muu = True
    veeris = [r for r in regs if r["klass"].startswith("veeris")]
    naaber = [r for r in regs if r["klass"] == "naaberleht"]
    if sees_muu:
        return None, "ploki_sees_kahtlane"
    if not veeris and not naaber and not sees_num:
        return gt_read, "muutmata"
    if len(xml_read) != len(gt_read):
        return None, "ridade_arv_erineb"
    gt_of = {(id(r), i): gt_read[k] for k, (r, i) in enumerate(xml_read)}

    # Põhiread lugemisjärjekorras regiooni kaupa; veerise read tõstetakse peremehe sisse.
    sisu = {}
    for r in regs:
        if r["klass"].startswith("veeris") or r["klass"] == "naaberleht":
            continue
        sisu[id(r)] = [(yk(bb), gt_of[(id(r), i)]) for i, (_, bb) in enumerate(r["ridade_bbox"])
                       if (id(r), i) not in sees_num]
    if not plokid:
        return None, "plokk_puudub"

    def lisa(host, y, t):
        rr = sisu[id(host)]
        k = next((j for j, (yy, _) in enumerate(rr) if yy is not None and y is not None and yy > y), len(rr))
        rr.insert(k, (y, f"<m>{t}</m>"))

    for blk in plokid:                               # ploki-sisesed numbrid: oma plokki, oma kõrgusele
        for i in sorted(i for (bid, i) in sees_num if bid == id(blk)):
            lisa(blk, yk(blk["ridade_bbox"][i][1]), gt_of[(id(blk), i)])
    for m in veeris:                                 # veerise regioon: suurim y-kattuvus, siis lähim x
        my0, my1 = m["bbox"][1], m["bbox"][3]; mx = (m["bbox"][0] + m["bbox"][2]) / 2
        host = max(plokid, key=lambda b: (max(0, min(my1, b["bbox"][3]) - max(my0, b["bbox"][1])),
                                          -abs(mx - (b["bbox"][0] + b["bbox"][2]) / 2)))
        for i, (_, bb) in sorted(enumerate(m["ridade_bbox"]), key=lambda x: yk(x[1][1]) or 0):
            lisa(host, yk(bb), gt_of[(id(m), i)])
    read = [t for r in regs if id(r) in sisu for _, t in sisu[id(r)]]
    return read, ("naaberleht_välja" if naaber and not veeris and not sees_num else "veeris_m")


def main():
    with open(BASE / "metadata.csv", encoding="utf-8", newline="") as f:
        rd = csv.reader(f); header = next(rd); rows = [list(r) for r in rd]
    # allikas → norm(tekst) → KÕIK rea indeksid: v5m-is on sama lehe tekst mitmes reas
    # (svea 08342 jt, 14.10) — muudatus peab minema kõigile, muidu jääb duplikaat vanaks.
    idx = defaultdict(lambda: defaultdict(list))
    for k, r in enumerate(rows):
        idx[r[2]][vt.norm(r[1])[:400]].append(k)
    valja = set(); muudetud = {}
    nimed = defaultdict(Counter)
    for allikas, stem, xs in vt.xmlid():
        a = vt.analyysi(xs)
        if not a:
            continue
        ks = [k for k in idx[allikas].get(vt.norm(a["tekst"])[:400], []) if k not in muudetud and k not in valja]
        for k in ks:
            read, pohjus = uus_tekst(a, rows[k][1].split("\n"))
            nimed[allikas][pohjus] += 1
            if read is None:
                valja.add(k)
            elif pohjus != "muutmata":
                muudetud[k] = "\n".join(read)
    for k, t in muudetud.items():
        rows[k][1] = t
    out = [r for k, r in enumerate(rows) if k not in valja]
    valja_nimed = {os.path.basename(rows[k][0]) for k in valja}

    print(f"v5m {len(rows)} → v6 {len(out)} (välja {len(valja)}, muudetud {len(muudetud)})")
    for al, c in sorted(nimed.items()):
        print(f"  {al:28s} {dict(c)}")
    mrd = Counter(r[2] for r in out for _ in range(r[1].count("<m>")))
    print(f"<m>-ridu allika kaupa: {dict(mrd.most_common())}  kokku {sum(mrd.values())}")
    if DRY:
        for k in list(muudetud)[:3]:
            print("\n---", rows[k][0]); print(rows[k][1][:900])
        return

    if NEW.exists():
        shutil.rmtree(NEW)
    (NEW / "images").mkdir(parents=True)
    for r in out:
        os.link(BASE / r[0], NEW / r[0])
    with open(NEW / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(out)
    if (BASE / "projektid.csv").exists():
        with open(BASE / "projektid.csv", encoding="utf-8") as f, open(NEW / "projektid.csv", "w", encoding="utf-8") as g:
            for l in f:
                if os.path.basename(l.split(",")[0]) not in valja_nimed:
                    g.write(l)
    ho = (BASE / "holdout.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    ho_valja = [l for l in ho if l.strip() and not l.startswith("#")
                and os.path.basename(l.strip().split("\t")[-1]) in valja_nimed]
    with open(NEW / "holdout.txt", "w", encoding="utf-8") as f:
        f.writelines(l for l in ho if l not in ho_valja)
        f.write(f"# v6: −{len(ho_valja)} lehte (veeris ebaselge / laotus ühes regioonis)\n")
    (NEW / "SOURCE.txt").write_text(
        (BASE / "SOURCE.txt").read_text(encoding="utf-8")
        + f"\nehitatud: {datetime.now():%Y-%m-%dT%H:%M:%S}  scripts/build_kurrent_v6.py\n"
        f"v5m {len(rows)} → {len(out)}  välja {len(valja)}, veeris <m>-iks {len(muudetud)} lehel; "
        f"{dict((a, dict(c)) for a, c in nimed.items())}  holdout −{len(ho_valja)}\n"
        "PAGE XML veeris → <m> (Riksarkivet 7 + senatsprotokolle), tuvastus scripts/veeris_tuvastus.py\n",
        encoding="utf-8")
    print(f"valmis: {NEW}  (holdout −{len(ho_valja)})")


if __name__ == "__main__":
    main()
