#!/usr/bin/env python3
"""Kurrendi andmestik v6 (2026-10-14) = v5m + veeris <m>-ina PAGE XML-i allikates.

v5-s kandis <m>-i ainult Leibniz + DTA; PAGE XML-i allikates oli veeris tavaridadena
lugemisjärjekorras (sageli lehe alguses/lõpus) → mudel pani kantseleikäte veerise
märgendita lõppu (VUTT agnp5i, 14.10). v6 teeb Riksarkiveti 7 allikas ja Tübingeni
senatsprotokollides veerise `<m>…</m>`-ridadeks omal kohal (iga füüsiline rida
omaette, VUTT ADR 0009; paigutus nagu v5 Leibnizil: peremeesploki ette esimese
põhirea ette, mille keskpunkt on temast allpool).

Allikad: Riksarkivet 7, senatsprotokolle, hanse XVI/XVII, kurrent-xix (xix_read_*); viimased
kolm HF-ist uuesti scripts/fetch_raw_xml_v6.py-ga (ainult XML).
Tuvastus: scripts/veeris_tuvastus.py (geomeetria; reeglid kasutaja kontrolliga 14.10).
Reeglid (SÜMMEETRILISED — kasutaja 14.10: ühe poole eelistamine õpetaks teise poole
veerise vahelejätmist):
  - veeris eraldi regioonina → <m>, mõlemal pool;
  - veeris põhiploki SEES → <m> ainult kui rida on number („11.", „5");
  - muu ploki-sisene kandidaat (taandrida, ületäie rea sõna, tabelirida või päris
    veeris — ei erista) → KOGU LEHT VÄLJA, mõlemal pool;
  - kahe lehekülje laotus ühes regioonis → leht välja;
  - veerise tekst > 0,5 põhiteksti pikkusest (rööptekst) või ≥ 15 veeriserida, > 40 % numbrid (tabel) → välja;
  - naaberlehe/dorsaali tekst → read GT-st välja (pildil näha, aga ei kuulu lehele;
    sama reegel mis VUTT Gemini juhises, PR #600).
Tekst võetakse v5m GT-st REAINDEKSI järgi (makron, ſ→s jm v4/v5 teisendused jäävad);
kui XML-i ja GT ridade arv ei klapi ja lehel on midagi muuta → leht välja.

Pildid hardlink-itud. Sisendkaustu ei muudeta. Holdout = v5m holdout miinus väljajäetud.

  venv/bin/python scripts/build_kurrent_v6.py [--dry-run]
"""
import csv, difflib, hashlib, os, re, shutil, sys
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


LISAMARK = "+×#^‸÷/"
ILN = Counter()
ILN_LOG = []      # --iln-naited: (B uus tekst, L ja B bbox) visuaalseks kontrolliks


def lisa_add(rida, lisad):
    """Interlineaarsed lisandused rea B sisse <add>-ina (ADR 0067). lisad = [(tekst, x, (bx0, bx1), sõnade_x)].
    Koht: Riksarkiveti sõnakoordinaadid (kui sõnu on sama palju kui GT-s), muidu x-i osakaal reast →
    lähim sõnapiir. Kui lisandus algab lisamärgiga (või B-s on täpselt üks eraldiseisev märk) ja B-s on
    sama märk eraldi sõnana, läheb ta märgi asemele. Valimis (30, 14.10) ~25 õiget; teadaolev viga:
    eelmise rea ületäie sõna („werden dick / verrührt.") ja päiseread („a Parte / Praes. / a Parte")."""
    tok = rida.split()
    kohad = []
    for t, x, (bx0, bx1), sonad in lisad:
        mark = t[0] if t[0] in LISAMARK else None
        t = t.lstrip(LISAMARK).strip()
        if not mark:                                     # hanse „delen + dath": märk ainult B-s
            margid = [w for w in tok if w and w in LISAMARK]
            if len(margid) == 1:
                mark = margid[0]
        if mark and mark in tok:
            k = tok.index(mark); tok[k] = None; kohad.append((k, x, t)); ILN["märgi_kohale"] += 1
            continue
        if sonad and len(sonad) == len(tok):
            k = sum(1 for s in sonad if s < x); ILN["sõnakoord"] += 1
        else:
            siht = max(0.0, min(1.0, (x - bx0) / max(bx1 - bx0, 1))) * len(" ".join(w or "" for w in tok))
            algused, pos = [], 0
            for w in tok:
                algused.append(pos); pos += len(w or "") + 1
            algused.append(len(" ".join(w or "" for w in tok)))
            k = min(range(len(algused)), key=lambda j: abs(algused[j] - siht)); ILN["osakaal"] += 1
        kohad.append((k, x, t))
    for k, _, t in sorted(kohad, reverse=True):
        if k < len(tok) and tok[k] is None:
            tok[k] = f"<add>{t}</add>"
        else:
            tok.insert(k, f"<add>{t}</add>")
    return " ".join(w for w in tok if w is not None)


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
    # Rööptekst (hanse17 237-0012/0013: teine veerg kordab põhiteksti). Ridade arv EI erista (rüütelkonna
    # protokollide teemaveerud on 12–30 rida, päris veeris), sõnade/märkide kattuvus ka mitte (veeris kordab
    # nimesid/märksõnu, 0,7–0,8). Eristab PIKKUS: rööptekst ~ põhiteksti pikkune (0,77–0,8), veeris ≤ ~0,45.
    pikkus = lambda ts: sum(len(re.sub(r"\W", "", t)) for t in ts)
    m_pikk = pikkus(t for r in veeris for t, _ in r["ridade_bbox"])
    p_pikk = pikkus(t for r in plokid for t, _ in r["ridade_bbox"])
    if m_pikk > 0.5 * max(p_pikk, 1):
        return None, "rööptekst"
    # Tabel (xix_read_1900 075047 metsatabelid): palju „veerist", enamus numbrid.
    vread = [t for r in veeris for t, _ in r["ridade_bbox"]] + [None] * len(sees_num)
    num = sum(1 for t in vread if t is None or NUMBER.fullmatch(t.replace("„", "").replace('"', "")))
    if len(vread) >= 15 and num > 0.4 * len(vread):
        return None, "tabel"
    # Interlineaarsed lisandused (svea 08342 „vices, / förrätta / så skall") → <add> alumise rea B sisse.
    iln = {}
    for blk in plokid:
        kand, jaetud = vt.interlineaarsed(blk, a["ns"])
        ILN["kahtlane_jäi_reaks"] += jaetud
        for li, bi, x, bx, sonad in kand:
            if (id(blk), li) not in sees_num and (id(blk), bi) not in sees_num:
                iln[(id(blk), li)] = (bi, x, bx, sonad)
    if not veeris and not naaber and not sees_num and not iln:
        return gt_read, "muutmata"
    if len(xml_read) != len(gt_read):
        return None, "ridade_arv_erineb"
    gt_of = {(id(r), i): gt_read[k] for k, (r, i) in enumerate(xml_read)}
    b_lisad = defaultdict(list)
    for (bid, li), (bi, x, bx, sonad) in iln.items():
        b_lisad[(bid, bi)].append((gt_of[(bid, li)], x, bx, sonad))
    for key, lisad in b_lisad.items():
        if key in iln:                     # B on ise lisandus (ahel) — ei puutu
            continue
        gt_of[key] = lisa_add(gt_of[key], lisad); ILN["B_ridu"] += 1
        blk = next(r for r in regs if id(r) == key[0])
        ILN_LOG.append((gt_of[key], [blk["ridade_bbox"][key[1]][1]]
                        + [blk["ridade_bbox"][li][1] for (bid, li), v in iln.items() if bid == key[0] and v[0] == key[1]]))

    # Põhiread lugemisjärjekorras regiooni kaupa; veerise read tõstetakse peremehe sisse.
    sisu = {}
    for r in regs:
        if r["klass"].startswith("veeris") or r["klass"] == "naaberleht":
            continue
        sisu[id(r)] = [(yk(bb), gt_of[(id(r), i)]) for i, (_, bb) in enumerate(r["ridade_bbox"])
                       if (id(r), i) not in sees_num
                       and not ((id(r), i) in iln and (id(r), iln[(id(r), i)][0]) not in iln)]
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
    if not veeris and not naaber and not sees_num:
        return read, "interlineaar_add"
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
        ks_koik = idx[allikas].get(vt.norm(a["tekst"])[:400], [])
        nimed[allikas]["xml_seotud" if ks_koik else "xml_sidumata"] += 1
        ks = [k for k in ks_koik if k not in muudetud and k not in valja]
        for k in ks:
            n0 = len(ILN_LOG)
            read, pohjus = uus_tekst(a, rows[k][1].split("\n"))
            for j in range(n0, len(ILN_LOG)):
                ILN_LOG[j] = (rows[k][0], rows[k][2]) + ILN_LOG[j]
            nimed[allikas][pohjus] += 1
            if read is None:
                valja.add(k)
            elif pohjus != "muutmata":
                muudetud[k] = "\n".join(read)
    for k, t in muudetud.items():
        rows[k][1] = t
    # Identse PILDIGA read (v5m: 142 gruppi / 437 rida, 14.10). Sama tekst → üks alles. Erinev tekst →
    # KÕIK treeningliikmed välja: üks versioon on vigane (aaeb part-2 „…_4428135_…": „Mille 1134" vs
    # „juillet 1579", „uyt" vs „vingt"), aga kumb, ei ole automaatselt otsustatav — lühim ei ole
    # diplomaatiline. Holdouti liige jääb alati alles (treeningkaksik välja = leke kaob).
    ho_nimed = {os.path.basename(l.strip().split("\t")[-1]) for l in open(BASE / "holdout.txt", encoding="utf-8")
                if l.strip() and not l.startswith("#")}
    grupid = defaultdict(list)
    for k, r in enumerate(rows):
        if k not in valja:
            grupid[hashlib.md5((BASE / r[0]).read_bytes()).hexdigest()].append(k)
    dup = Counter(); dup_naited = []
    for ks in grupid.values():
        if len(ks) < 2:
            continue
        t = [vt.norm(rows[k][1]) for k in ks]
        sama = min(difflib.SequenceMatcher(None, t[0][:2000], x[:2000]).ratio() for x in t[1:]) > 0.95
        ho = [k for k in ks if os.path.basename(rows[k][0]) in ho_nimed]
        jaab = ho[0] if ho else (ks[0] if sama else None)
        for k in ks:
            if k != jaab and os.path.basename(rows[k][0]) not in ho_nimed:
                valja.add(k); dup["sama_tekst" if sama else "erinev_tekst"] += 1
        if not sama and len(dup_naited) < 4:
            dup_naited.append([(os.path.basename(rows[k][0])[:30], "JÄÄB" if k == jaab else "välja",
                                rows[k][1][:70].replace("\n", " / ")) for k in ks])
    print(f"duplikaadid (identne pilt) välja: {dict(dup)}")
    for n in dup_naited:
        for x in n:
            print("   ", x)
        print()
    out = [r for k, r in enumerate(rows) if k not in valja]
    valja_nimed = {os.path.basename(rows[k][0]) for k in valja}

    print(f"v5m {len(rows)} → v6 {len(out)} (välja {len(valja)}, muudetud {len(muudetud)})")
    for al, c in sorted(nimed.items()):
        print(f"  {al:28s} {dict(c)}")
    mrd = Counter(r[2] for r in out for _ in range(r[1].count("<m>")))
    print(f"interlineaar: {dict(ILN)}")
    if "--iln-naited" in sys.argv:
        import json, random
        random.seed(2)
        json.dump(random.sample(ILN_LOG, 30), open("/tmp/iln_naited.json", "w"), ensure_ascii=False)
    ard = Counter(r[2] for r in out for _ in range(r[1].count("<add>")))
    print(f"<add> allika kaupa: {dict(ard.most_common())}  kokku {sum(ard.values())}")
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
        "PAGE XML veeris → <m> (Riksarkivet 7, senatsprotokolle, hanse XVI/XVII, kurrent-xix), tuvastus scripts/veeris_tuvastus.py\n"
        f"identse pildiga duplikaadid välja: {dict(dup)} (erinev tekst → kõik treeningliikmed välja)\n",
        encoding="utf-8")
    print(f"valmis: {NEW}  (holdout −{len(ho_valja)})")


if __name__ == "__main__":
    main()
