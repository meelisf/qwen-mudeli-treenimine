#!/usr/bin/env python3
"""Veerise regioonide tuvastus PAGE XML-ist geomeetria järgi (Kurrent v6 eeltöö).

Riksarkiveti ja ubtue-gt regioonid on enamasti tüübita, seega veeris tuvastatakse
asukoha järgi: põhiveerg = kõige rohkemate ridadega regioon; veeris = regioon, mis
jääb põhiveerust horisontaalselt (peaaegu) välja ja on sellest kitsam. Tüüp
`marginalia` loetakse veeriseks, `page-number` ja 1-realised ≤4-märgilised
numbrid jäetakse kõrvale (leheküljenumber / püüdsõna, mitte veeris).

XML seotakse treeningpildiga TEKSTI järgi (pildinimed on 60 märgini lõigatud ega
ole unikaalsed): XML-i lugemisjärjekorras tekst vs metadata.csv GT.

  venv/bin/python veeris_tuvastus.py            # statistika + näited → data/v6_veeris/
"""
import csv, json, re, tarfile, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

csv.field_size_limit(10 ** 8)
RAW = Path("data/raw_xml")
GT_CSV = Path("data/kurrent_v5m/metadata.csv")
IMG = Path("data/kurrent_v5m/images")
OUT = Path("data/v6_veeris")
NAITEID = 8          # näidet allika kohta (pildiga)
VAIKE_KATTUVUS = 0.3  # veeris: ≤30 % oma laiusest kattub põhiveeruga
KITSAS = 0.6          # veeris: laius ≤ 60 % põhiveerust (laiem = teine veerg)

RIKS = {   # HF nimi → v5m allikas
    "bergskollegium_advokatfiskalskontoret_seg": "bergskollegium_adv_seg",
    "bergskollegium_relationer_och_skrivelser_seg": "bergskollegium_rel_seg",
    "gota_hovratt_seg": "gota_hovratt_seg",
    "jonkopings_radhusratts_och_magistrat_seg": "jonkopings_seg",
    "krigshovrattens_dombocker_seg": "krigshovrattens_seg",
    "svea_hovratt_seg": "svea_hovratt_seg",
    "trolldomskommissionen_seg": "trolldomskommissionen_seg",
}


def ns_of(root):
    m = re.match(r"\{(.*)\}", root.tag)
    return f"{{{m.group(1)}}}" if m else ""


def bbox(el, ns):
    c = el.find(f"{ns}Coords")
    if c is None:
        return None
    if c.get("points"):
        pts = [tuple(map(int, map(float, p.split(",")))) for p in c.get("points").split()]
    else:
        pts = [(int(p.get("x")), int(p.get("y"))) for p in c.findall(f"{ns}Point")]
    if not pts:
        return None
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def tyyp(r):
    t = r.get("type") or ""
    m = re.search(r"structure\s*\{[^}]*type:([\w-]+)", r.get("custom") or "")
    return (m.group(1) if m else t).lower()


def read_lines(r, ns):
    out = []
    for line in r.findall(f"{ns}TextLine"):
        te = line.find(f"{ns}TextEquiv")
        uc = te.find(f"{ns}Unicode") if te is not None else None
        if uc is not None and uc.text and uc.text.strip():
            out.append((uc.text.strip(), bbox(line, ns)))
    return out


def analyysi(xml_str):
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None
    ns = ns_of(root)
    page = root.find(f"{ns}Page")
    if page is None:
        return None
    ro = [x.get("regionRef") for x in page.iter(f"{ns}RegionRefIndexed")]
    regs = {r.get("id"): r for r in page.iter(f"{ns}TextRegion")}
    jarjestus = [regs[i] for i in ro if i in regs] + [r for i, r in regs.items() if i not in ro]
    info = []
    for r in jarjestus:
        b = bbox(r, ns); ls = read_lines(r, ns)
        if not ls:
            continue
        if b is None:
            bs = [l[1] for l in ls if l[1]]
            if not bs:
                continue
            b = (min(x[0] for x in bs), min(x[1] for x in bs), max(x[2] for x in bs), max(x[3] for x in bs))
        info.append({"tyyp": tyyp(r), "bbox": b, "read": [l[0] for l in ls], "ridade_bbox": ls, "el": r})
    if not info:
        return None
    tekst = "\n".join(l for r in info for l in r["read"])
    pohi = max(info, key=lambda r: (len(r["read"]), r["bbox"][2] - r["bbox"][0]))
    pw = max(pohi["bbox"][2] - pohi["bbox"][0], 1)
    try:
        lehe_laius, lehe_korgus = int(page.get("imageWidth") or 0), int(page.get("imageHeight") or 0)
    except ValueError:
        lehe_laius = lehe_korgus = 0
    def laius(r): return max(r["bbox"][2] - r["bbox"][0], 1)
    def cx(r): return (r["bbox"][0] + r["bbox"][2]) / 2
    def cy(r): return (r["bbox"][1] + r["bbox"][3]) / 2
    # Põhiplokid: kõik laiad regioonid (kahe lehekülje laotusel kaks). Number ei ole kunagi plokk.
    plokid = [r for r in info if r is pohi or (laius(r) > KITSAS * pw and len(r["read"]) > 1)]
    for blk in plokid:
        rb = [bb for _, bb in blk["ridade_bbox"] if bb]
        if len(rb) >= 5:
            x0s = sorted(bb[0] for bb in rb); x1s = sorted(bb[2] for bb in rb)
            blk["tuum"] = (x0s[len(x0s) // 2], x1s[len(x1s) // 2])
        else:
            blk["tuum"] = (blk["bbox"][0], blk["bbox"][2])
    for r in info:
        if r is pohi:
            r["klass"] = "põhi"; continue
        if r in plokid:
            r["klass"] = "teine_plokk"; continue
        if r["tyyp"] in ("page-number", "catch-word", "signature-mark"):
            r["klass"] = "number"; continue
        numbriline = len(r["read"]) == 1 and re.fullmatch(r"[\d\s.,:/]{1,6}", r["read"][0]) is not None
        # Tekst kõigi plokkide kohal = päis (jooksev päis, „Tabula 1."); number seal jääb veeriseks (svea „1020.").
        if not numbriline and r["bbox"][3] <= min(b["bbox"][1] for b in plokid) + 20:
            r["klass"] = "päis"; continue
        # Kas keskpunkt on mõne ploki x-vahemikus? Siis päis/jalus (ploki kohal/all) või põhi_lisa.
        sees = [b for b in plokid if b["tuum"][0] <= cx(r) <= b["tuum"][1]]
        if sees and r["tyyp"] != "marginalia":
            b = sees[0]
            r["klass"] = ("number" if numbriline else
                          "päis" if (cy(r) < b["bbox"][1] or cy(r) > b["bbox"][3]) else "põhi_lisa")
            continue
        # Keskpunkt plokkidest väljas: veeris või naaberlehe tekst (kaugel lähimast plokiservast).
        b = min(plokid, key=lambda b: min(abs(cx(r) - b["tuum"][0]), abs(cx(r) - b["tuum"][1])))
        vasak = cx(r) < (b["tuum"][0] + b["tuum"][1]) / 2
        r["kaugus"] = round(max(0, (b["tuum"][0] - cx(r)) if vasak else (cx(r) - b["tuum"][1])) / pw, 2)
        # y kattub MÕNE plokiga (kõrval, mitte kohal/all) — lähim x-i järgi võib olla lühike naaberleht
        yl = any(not (r["bbox"][3] < q["bbox"][1] or r["bbox"][1] > q["bbox"][3]) for q in plokid)
        if numbriline and r["kaugus"] > 0.5:
            r["klass"] = "number"         # laotuse teise lehe leheküljenumber („101.")
        elif r["tyyp"] != "marginalia" and (r["kaugus"] > 0.5 or (laius(r) > 0.3 * pw and r["kaugus"] > 0.28)):
            # vastaslehe/dorsaali tekst. Lai regioon on naaberleht alles > 0,28 kaugusel: svea laiad
            # nimeveerised on 0,30–0,41 laiad, aga 0,16–0,23 kaugusel (kasutaja kontroll 14.10).
            r["klass"] = "naaberleht"
        elif not yl and not numbriline and r["tyyp"] != "marginalia":
            r["klass"] = "päis" if r["bbox"][3] < min(q["bbox"][1] for q in plokid) else "jalus"
        else:
            r["klass"] = "veeris_" + ("vasak" if vasak else "parem")
    # Veeris PLOKI SEES (lähte-XML pani veerise read põhiregiooni, bergskoll 12729, trolldom 14725):
    # rida, mis jääb veeru mediaanvahemikust täielikult vasakule või paremale.
    for blk in plokid:
        rb = [(t, bb) for t, bb in blk["ridade_bbox"] if bb]
        if len(rb) < 5:
            continue
        x0s = sorted(bb[0] for _, bb in rb); x1s = sorted(bb[2] for _, bb in rb)
        m0, m1 = x0s[len(x0s) // 2], x1s[len(x1s) // 2]; tol = 0.05 * (m1 - m0)
        alumine = max(rb, key=lambda x: x[1][3])[1]   # ploki viimane rida: paremal = püüdsõna, mitte veeris
        indeks = {id(bb): i for i, (_, bb) in enumerate(blk["ridade_bbox"]) if bb}
        sees = [(t, "vasak" if bb[2] < m0 + tol else "parem") for t, bb in rb
                if (bb[2] < m0 + tol or bb[0] > m1 - tol) and not (bb is alumine and bb[0] > m1 - tol)]
        # v6 ehitaja jaoks: ploki ridade indeksid (blk["ridade_bbox"] järjekorras)
        blk["veeris_sees_idx"] = [indeks[id(bb)] for t, bb in rb
                                  if (bb[2] < m0 + tol or bb[0] > m1 - tol) and not (bb is alumine and bb[0] > m1 - tol)]
        if len(sees) > 0.3 * len(rb):
            blk["mitu_veergu"] = True   # 2 veergu/lehte ühes regioonis (svea 12251), mitte veeris
            sees = []; blk["veeris_sees_idx"] = []
        blk["veeris_sees"] = sees
    # Kahe lehekülje laotus ühes regioonis: rõhtpilt + üks plokk üle poole laiuse (svea 12251).
    laotus_yhes = (any(blk.get("mitu_veergu") for blk in plokid)
                   or (bool(lehe_laius) and lehe_laius > lehe_korgus and len(plokid) == 1
                       and pw > 0.6 * lehe_laius))
    return {"tekst": tekst, "regioonid": info, "laotus_yhes": laotus_yhes, "ns": ns,
            "veeris_lugemisjarjekorras": [i for i, r in enumerate(info) if r["klass"].startswith("veeris")]}


def _tekstiread(r_el, ns):
    """Regiooni TextLine-elemendid SAMA filtriga mis read_lines (indeksid ühtivad ridade_bbox-iga)."""
    out = []
    for line in r_el.findall(f"{ns}TextLine"):
        te = line.find(f"{ns}TextEquiv")
        uc = te.find(f"{ns}Unicode") if te is not None else None
        if uc is not None and uc.text and uc.text.strip():
            out.append(line)
    return out


def _baasjoon(line, ns):
    b = line.find(f"{ns}Baseline")
    if b is not None and b.get("points"):
        ys = [float(p.split(",")[1]) for p in b.get("points").split()]
        return sum(ys) / len(ys)
    bb = bbox(line, ns)
    return bb[3] if bb else None


ILN_NUMBER = re.compile(r"[\d\s.,:/§°o]{1,6}")


def interlineaarsed(blk, ns):
    """Ploki interlineaarsed lisandused (v6, 14.10): [(L_idx, B_idx, x_keskpunkt, B sõnade x-keskpunktid | None)].

    L = lühike rida (< 0,5 ploki rea mediaanlaiusest), mille baasjoon on kahe JÄRJESTIKUSE põhirea A ja B
    vahel (B−A < 1,4 × reasamm) ja mis kattub x-is mõlemaga; lisandus kuulub alumisse ritta B.
    Juhuvalimis (24) 14 õiget / 8 vale — valed välja reeglitega: tsentreeritud pealkiri („VORTRAG / des"),
    nimekiri/retsept (A, L, B algavad samast x-ist või kõik numbriga, „1 Ei"), number/ordinaal („1:o").
    Tagastab ka vahele jäetud kandidaatide arvu (teine väärtus)."""
    els = _tekstiread(blk["el"], ns)
    L = []
    for i, (el, (t, bb)) in enumerate(zip(els, blk["ridade_bbox"])):
        y = _baasjoon(el, ns)
        if bb and y is not None:
            L.append(dict(i=i, t=t, x0=bb[0], x1=bb[2], y=y, el=el))
    if len(L) < 4:
        return [], 0
    mw = sorted(l["x1"] - l["x0"] for l in L)[len(L) // 2]
    pikad = sorted([l for l in L if l["x1"] - l["x0"] >= 0.6 * mw], key=lambda l: l["y"])
    sammud = sorted(b["y"] - a["y"] for a, b in zip(pikad, pikad[1:]) if b["y"] > a["y"])
    if len(pikad) < 3 or not sammud:
        return [], 0
    samm = sammud[len(sammud) // 2]
    cx = lambda l: (l["x0"] + l["x1"]) / 2
    kattub = lambda p, l: min(p["x1"], l["x1"]) - max(p["x0"], l["x0"]) > 0.5 * (l["x1"] - l["x0"])
    out, jaetud = [], 0
    for l in L:
        if l["x1"] - l["x0"] >= 0.5 * mw:
            continue
        yle = [p for p in pikad if p["y"] < l["y"] - 0.15 * samm and kattub(p, l)]
        alla = [p for p in pikad if p["y"] > l["y"] + 0.15 * samm and kattub(p, l)]
        if not (yle and alla) or alla[0]["y"] - yle[-1]["y"] >= 1.4 * samm:
            continue
        a, b = yle[-1], alla[0]
        tol = 0.05 * mw
        if (ILN_NUMBER.fullmatch(l["t"])                                           # „1:o", „5"
                or len(re.findall(r"[^\W\d_]", l["t"])) < 2                          # „╕" (senats): märk
                or (abs(l["x0"] - a["x0"]) < tol and abs(l["x0"] - b["x0"]) < tol)  # nimekiri
                or all(re.match(r"\s*\d", x["t"]) for x in (a, l, b))               # retsept, tabel
                or (abs(cx(l) - cx(a)) < tol and abs(cx(a) - cx(b)) < 2 * tol)       # tsentreeritud pealkiri
                or len(a["t"].split()) < 3 or len(b["t"].split()) < 3):              # pealkiri, aadress
            jaetud += 1
            continue
        sonad = []
        for w in b["el"].findall(f"{ns}Word"):
            wb = bbox(w, ns)
            if wb:
                sonad.append((wb[0] + wb[2]) / 2)
        out.append((l["i"], b["i"], cx(l), (b["x0"], b["x1"]), sorted(sonad) or None))
    return out, jaetud


def lipud(v):
    """Piirjuhud, mida geomeetria ei erista (kasutaja peab pildilt vaatama)."""
    out = set()
    for r in v:
        t = " ".join(r["read"])
        if re.fullmatch(r"[\d\s.,/:;\-–]+", t):
            out.add("ainult_numbrid")      # arhiivinumber, kuupäev, foliant
        if re.match(r"\s*[+×#^‸/\\]|\s*[a-z]\)", t):
            out.add("lisamärk")            # tõenäoliselt <add> (ADR 0067), mitte <m>
        if len(r["read"]) >= 8:
            out.add("pikk")                # võib olla veerg või lisatud lõik
    return sorted(out)


def norm(t):
    t = unicodedata.normalize("NFC", t).replace("ſ", "s")
    t = re.sub(r"</?[a-z]+\d*/?>", "", t)
    return re.sub(r"\W+", "", t.lower())


def xmlid():
    for hf, allikas in RIKS.items():
        for tgz in sorted((RAW / "riksarkivet").glob(f"datasets--Riksarkivet--{hf}/snapshots/*/data/page_xmls/*.tar.gz")):
            with tarfile.open(tgz) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".xml"):
                        yield allikas, Path(m.name).stem, tf.extractfile(m).read().decode("utf-8", "replace")
    for p in sorted((RAW / "ubtue-gt").rglob("*.xml")):
        if "Senatsprotokolle" in str(p):
            yield "senatsprotokolle", p.stem, p.read_text(encoding="utf-8", errors="replace")
    # v6 (14.10): HF-ist uuesti alla laaditud (scripts/fetch_raw_xml_v6.py)
    for kaust, allikas in (("hanse_xvi", "hanse_kurrent_xvi"), ("hanse_xvii", "hanse_kurrent_xvii")):
        for p in sorted((RAW / kaust).rglob("*.xml")):
            yield allikas, p.stem, p.read_text(encoding="utf-8", errors="replace")
    proj_allikas = xix_projektid()
    for p in sorted((RAW / "kurrent_xix").rglob("*.xml")):
        a = proj_allikas.get(p.parent.name)
        if a:
            yield a, p.stem, p.read_text(encoding="utf-8", errors="replace")


def xix_projektid():
    """kurrent-xix projekt → v5m allikas (xix_read_<periood>), v5m projektid.csv + metadata järgi."""
    al = {r["failinimi"]: r["allikas"] for r in csv.DictReader(open(GT_CSV, encoding="utf-8"))}
    out = {}
    pr = GT_CSV.parent / "projektid.csv"
    if pr.exists():
        for r in csv.reader(open(pr, encoding="utf-8")):
            if len(r) > 1 and r[0] in al and al[r[0]].startswith(("xix_read", "kurrent_xix")):
                out[re.sub(r"[^\w\-]", "_", r[1])] = al[r[0]]
    return out


def main():
    OUT.mkdir(exist_ok=True); (OUT / "naited").mkdir(exist_ok=True)
    gt = defaultdict(dict)   # allikas → norm(tekst) → failinimi
    for r in csv.DictReader(open(GT_CSV, encoding="utf-8")):
        gt[r["allikas"]][norm(r["transkriptsioon"])[:400]] = r["failinimi"]
    stat = defaultdict(Counter); naited = defaultdict(list); koik = []
    for allikas, stem, xs in xmlid():
        a = analyysi(xs)
        s = stat[allikas]; s["xml"] += 1
        if a is None:
            s["parse_viga"] += 1; continue
        fail = gt[allikas].get(norm(a["tekst"])[:400])
        s["treeningus" if fail else "treeningust_väljas"] += 1
        k = Counter(r["klass"] for r in a["regioonid"])
        for kl, n in k.items():
            s["reg_" + kl] += n
        v = [r for r in a["regioonid"] if r["klass"].startswith("veeris")]
        sees = [x for r in a["regioonid"] for x in r.get("veeris_sees", [])]
        if sees:
            s["lk_veeris_sees"] += 1; s["veeris_sees_ridu"] += len(sees)
            v = v + [{"klass": "veeris_sees_" + k, "read": [t], "bbox": (0, 0, 0, 0)} for t, k in sees]
        if k["teine_plokk"]:
            s["lk_teine_veerg"] += 1
        if a["laotus_yhes"]:
            s["lk_laotus_yhes"] += 1
        if v:
            s["lk_veerisega"] += 1
            s["veeris_ridu"] += sum(len(r["read"]) for r in v)
            if fail: s["lk_veerisega_treeningus"] += 1
            # kus veeris lugemisjärjekorras on: alguses / lõpus / vahel
            idx = a["veeris_lugemisjarjekorras"]; n = len(a["regioonid"])
            # veeris ainult ploki sees = juba põhiteksti ridade vahel (lugemisjärjekorras „vahel")
            s["asukoht_" + ("vahel" if not idx else "algus" if idx[0] == 0 else "lõpp" if idx[-1] == n - 1 else "vahel")] += 1
            kirje = {"allikas": allikas, "xml": stem, "pilt": fail,
                     "veeris": [{"klass": r["klass"], "kaugus": r.get("kaugus"), "ridu": len(r["read"]),
                                 "read": r["read"][:4]} for r in v],
                     "lipud": lipud(v) + (["laotus_yhes"] if a["laotus_yhes"] else [])}
            koik.append(kirje)
            if fail and len(naited[allikas]) < NAITEID:
                naited[allikas].append(kirje)
                joonista(fail, a, allikas, len(naited[allikas]))
    json.dump(koik, open(OUT / "veerisega_lehed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({k: dict(v) for k, v in stat.items()}, open(OUT / "statistika.json", "w"), ensure_ascii=False, indent=1)
    json.dump(naited, open(OUT / "naited.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'allikas':28s} {'xml':>5s} {'treen':>5s} {'veer.lk':>7s} {'veer.rd':>7s} {'algus':>5s} {'lõpp':>5s} {'vahel':>5s} {'2.plokk':>7s} {'number':>6s} {'päis':>5s} {'naaber':>6s} {'1laot':>5s} {'sees.lk':>7s} {'sees.rd':>7s}")
    for al, s in sorted(stat.items()):
        print(f"{al:28s} {s['xml']:5d} {s['treeningus']:5d} {s['lk_veerisega']:7d} {s['veeris_ridu']:7d} "
              f"{s['asukoht_algus']:5d} {s['asukoht_lõpp']:5d} {s['asukoht_vahel']:5d} {s['lk_teine_veerg']:7d} {s['reg_number']:6d} {s['reg_päis']:5d} {s['reg_naaberleht']:6d} {s['lk_laotus_yhes']:5d} {s['lk_veeris_sees']:7d} {s['veeris_sees_ridu']:7d}")


def joonista(fail, a, allikas, nr):
    """Pilt kastidega: põhi sinine, veeris punane, number hall, teine veerg oranž."""
    p = IMG / Path(fail).name
    if not p.exists():
        return
    im = Image.open(p).convert("RGB")
    xs = [r["bbox"][2] for r in a["regioonid"]]; ys = [r["bbox"][3] for r in a["regioonid"]]
    # treeningpilt võib olla vähendatud — skaleeri XML-i koordinaadid pildi suuruseks (hinnang)
    sx = im.width / max(max(xs) * 1.05, im.width) if max(xs) > im.width else 1
    sy = im.height / max(max(ys) * 1.05, im.height) if max(ys) > im.height else 1
    s = min(sx, sy) if (sx < 1 or sy < 1) else 1
    d = ImageDraw.Draw(im)
    varv = {"põhi": "blue", "number": "gray", "teine_plokk": "orange", "põhi_lisa": "green", "päis": "purple", "jalus": "purple", "naaberleht": "black"}
    for r in a["regioonid"]:
        x0, y0, x1, y1 = [int(v * s) for v in r["bbox"]]
        d.rectangle([x0, y0, x1, y1], outline=varv.get(r["klass"], "red"), width=6)
    im.thumbnail((1600, 1600))
    im.save(OUT / "naited" / f"{allikas}_{nr:02d}.jpg", quality=85)


if __name__ == "__main__":
    main()
