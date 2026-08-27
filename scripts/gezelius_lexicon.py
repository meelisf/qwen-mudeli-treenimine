#!/usr/bin/env python3
"""
Gezeliuse Kreeka-Ladina leksikon: VUTT-i eksport ja pseudomärgendus

Lexicon on trükikorpuse kreeka selgroog: 447 lehte, millest 549 kogu
andmestiku 559-st kreekarikkast lehest. VUTT-is teda EI OLE (seal on ainult
Comenius-Ianua, 184 lk „Toores"). Sellepärast see eksport.

Kaks eraldi asja, mida see skript teeb – neid EI TOHI segamini ajada:

1. HOMOGLÜÜFIDE PARANDUS (`fix_homoglyphs`) – transkriptsiooniviga.
   Suurtähelistes kreeka märksõnades on ladina näoga tähed:
   `KNΊΣΣA` = ladina K, N, A + kreeka ΊΣΣ. Mõõdetud: 1472 sõna ehk 5,5 %
   kõigist kreekat sisaldavatest sõnadest. Ianuas 8 ehk 0,0 %.
   Parandus on ühene: ladina täht kreekat sisaldava sõna sees → kreeka vaste.
   See läheb ekspordi sisse, sest parandamata teksti VUTT-i tõstmine
   kinnistaks vea ground truth'i.

2. PSEUDOMÄRGENDUS (`pseudo_markup`) – SÜNTEETILINE ground truth.
   Reegel: kreeka lõikude vahel olev ladinatäheline lõik on kursiivis
   antiikvas → `<i>`. Struktuurne alus: Lexiconis on 81,8 % ridadest
   mõlema kirjaga, Ianuas 0,2 %.
   See EI lähe ekspordi sisse. Sünteetilise GT viga on süstemaatiline ja
   iseendaga kooskõlas – ta ei paista lossis ega andmeid sirvides. Enne
   kasutamist tuleb reegel valideerida käsitsi märgendatud lehtede vastu.

Käivitamine:
  python scripts/gezelius_lexicon.py --stats
  python scripts/gezelius_lexicon.py --export data/export/gezelius-lexicon
  python scripts/gezelius_lexicon.py --pseudo data/export/gezelius-lexicon-pseudo
  python scripts/gezelius_lexicon.py --valideeri KÄSITSI_DIR PSEUDO_DIR
"""

import csv
import json
import re
import shutil
import sys
from pathlib import Path

csv.field_size_limit(10 ** 7)

DATA_CSV = Path("data/lehekyljed/metadata.csv")
IMAGES = Path("data/lehekyljed/images")
TEOS = "Lexicon_exact"

#: Ladina tähed, mille kreeka vaste on visuaalselt identne. Ainult neid
#: teisendatakse, ja ainult kreekat sisaldava sõna sees.
HOMOGLYPHS = {
    "A": "Α", "B": "Β", "E": "Ε", "Z": "Ζ", "H": "Η", "I": "Ι", "K": "Κ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "Y": "Υ", "X": "Χ",
    "a": "α", "b": "β", "e": "ε", "k": "κ", "o": "ο", "p": "ρ", "v": "ν",
    "x": "χ",
}

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
GREEK_RUN = re.compile(r"([Ͱ-Ͽἀ-῿]+)")
LATIN = re.compile(r"[A-Za-zÀ-ÿſ]")
WORD = re.compile(r"[A-Za-zÀ-ÿſͰ-Ͽἀ-῿]+")
PAGE_NR = re.compile(r"^(\d+)_Gezelius_" + TEOS)


def fix_homoglyphs(text: str) -> str:
    """Ladina näoga tähed kreekat sisaldavate sõnade sees → kreeka vasted.

    KNΊΣΣA → ΚΝΊΣΣΑ.  Puhtladina sõnu ei puudutata.
    """
    def one(mo):
        w = mo.group(0)
        if not GREEK.search(w):
            return w
        return "".join(HOMOGLYPHS.get(c, c) for c in w)
    return WORD.sub(one, text)


#: Ladina täht, sh see, mis meie enda paranduse järel tekib.
_LAT = "A-Za-zÀ-ÿſæœÆŒ"

#: Sõnad, kus `ae`/`oe` EI ole ligatuur. Kontrollitud originaalpildilt:
#: `coerceri` (lk 0155) on trükitud lahku, sest co-+arceo ei ole diftong;
#: `Israelitarum` (lk 0252) samuti. Kõik ülejäänud kontrollitud juhud
#: (cœtus lk 0026, proœmium lk 0283, Ægypti lk 0303, GRÆCO lk 0005) on
#: ligatuuriga, ka suurtähtedes.
NO_LIGATURE = re.compile(r"^(coerc|Israel)", re.I)

_LIGATURES = [("ae", "æ"), ("oe", "œ"), ("Ae", "Æ"), ("AE", "Æ"),
              ("Oe", "Œ"), ("OE", "Œ")]


def fix_ligatures(text: str) -> str:
    """`ae`/`oe` → `æ`/`œ` puhtladina sõnades (ka suurtähtedes)."""
    def one(mo):
        w = mo.group(0)
        if GREEK.search(w) or NO_LIGATURE.match(w):
            return w
        for a, b in _LIGATURES:
            w = w.replace(a, b)
        return w
    return WORD.sub(one, text)


#: Sõna, mille sees `s` on pikk. `¬` (reamurre) EI lõpeta sõna – „omis¬ / si"
#: keskel olev s on ikka pikk.
_S_WORD = re.compile(f"[{_LAT}]+¬?")


def fix_long_s(text: str) -> str:
    """`s` → `ſ` sõna alguses ja keskel; sõna lõpus jääb `s`.

    Antiikvakursiivi konventsioon, mille originaal järgib läbivalt:
    ſylveſtris, poſſeſſio, aſſentatio. Suurtähelist `S`-i see ei puuduta.
    """
    def one(mo):
        w = mo.group(0)
        if GREEK.search(w):
            return w
        # Viimane täht jääb lühikeseks – v.a kui sõna murdub reavahetusel
        # (lk 0014 „omiſ¬ / ſi"), sest siis ei ole see sõna lõpp.
        keep = -1 if w.endswith("¬") else len(w) - 1
        return "".join("ſ" if c == "s" and i != keep else c
                       for i, c in enumerate(w))
    return _S_WORD.sub(one, text)


def _is_running_head(line: str) -> bool:
    """Lehepäis: lühike rida numbriga, nt „ΚΝ ΚΟ   193".

    Osal lehtedel on leheküljenumber servast maha lõigatud või lugemata
    (lk 0195 „KA", 0416 „XE", 0021 „AB ΑΓ") – siis tunneme päise ära selle
    järgi, et rida koosneb ainult tähestikumarkeritest.
    """
    line = line.strip()
    if not line or len(line) >= 40:
        return False
    if re.search(r"\d", line):
        return True
    return bool(re.fullmatch(r"[A-ZΆ-Ϋἀ-῿ ]{2,12}", line))


#: Lehepäises on kreeka tähestikumarkerid (ΒΛ, ΕΓ, ΚΛ, ΜΗ, ΣΑ, ΦΑ). Osa neist
#: on transkribeeritud puhtladina tähtedega (OP, KA, BA…) ja siis ei näe neid
#: `fix_homoglyphs`, sest sõnas EI OLE ühtegi kreeka tähte. Mõõdetud: 198 sellist
#: sõna, neist 185 just lehepäises.
#: Piirang KAHELE tähele on tahtlik – päises on ka päris ladina sõnu
#: (lk 0440: „INDEX."), ja need on pikemad.
_HEAD_MARKER = re.compile(r"\b([A-Z]{2})\b")


def fix_running_head(line: str) -> str:
    """Kahetähelised puhtladina suurtähemarkerid lehepäises → kreeka."""
    def one(mo):
        w = mo.group(1)
        if all(c in HOMOGLYPHS for c in w):
            return "".join(HOMOGLYPHS[c] for c in w)
        return w
    return _HEAD_MARKER.sub(one, line)


#: Sõnastiku põhiosa. Enne seda on eeltekst, pärast INDEX – kummaski ei ole
#: „kreeka märksõna + kursiivne ladina gloss" struktuuri, ehk pseudomärgendus
#: sinna ei kehti. Vahemiku andis kasutaja pistelise kontrolli põhjal.
PSEUDO_RANGE = (21, 440)

#: Rida, mis koosneb ainult üksikust suurtähest (+ number/kirjavahemärk).
_SIGNATURE = re.compile(r"\s*[A-Z]\s*[\d.,)]*\s*")

#: Ladina sõna. `à` etümoloogiaviites („(à θέω)") on trükis püstkirjas, seega
#: ei loe seda kursiivilõigu sisuks – ilma selle erandita venib kursiiv üle
#: sulu kreeka sõnani.
_LATIN_WORD = re.compile(f"[{_LAT}]+")
_NOT_CONTENT = {"à"}

#: Kirjavahemärgid, mis kuuluvad kursiivi sisse: lause lõpupunkt ja
#: reamurde `¬`. Avasulg tõmmatakse sisse eraldi, vasakult.
_TAIL = re.compile(r"[.,;:!?¬)]+")


def _italic_span(seg: str):
    """(algus, lõpp) kursiiviga kaetavast osast, või None kui pole sisu.

    Piirid on võetud kasutaja käsitsi parandatud lehelt (VUTT lk 176 =
    `gezelius-lexicon-0180`): lõpupunkt ja `¬` jäävad `<i>` SISSE, avasulg
    samuti (`<i>(arbor quædam odora,) thya.</i>`), aga kreeka ees seisev
    etümoloogiasaba jääb välja (`<i>damnum: mulcta:</i> (à θέω)`).
    """
    words = [m for m in _LATIN_WORD.finditer(seg)
             if m.group(0) not in _NOT_CONTENT]
    if not words:
        return None
    a, b = words[0].start(), words[-1].end()
    if a and seg[a - 1] == "(":
        a -= 1
    m = _TAIL.match(seg, b)
    return a, m.end() if m else b


def pseudo_markup(text: str) -> str:
    """Märgib kreeka lõikude vahelised ladinatähelised lõigud `<i>`-ga.

    SÜNTEETILINE. Vt mooduli docstring'ut. Lehepäis jäetakse vahele.
    """
    read = text.split("\n")
    viimased = len(read) - 3  # poogna signatuur seisab lehe jalal
    out = []
    for idx, line in enumerate(read):
        if idx == 0 and _is_running_head(line):
            out.append(line)
            continue
        if _SIGNATURE.fullmatch(line):
            # Poogna signatuur lehe jalal (A, B, C… iga 16 lehe järel) on
            # trükis KURSIIVIS – vt lk 0125 „G 5". Sama mustriga rida lehe
            # ülaosas on aga sektsioonitäht (lk 0021 „A"), mis on püstkirjas.
            out.append(f"<i>{line.strip()}</i>" if idx >= viimased else line)
            continue
        parts = []
        for seg in GREEK_RUN.split(line):
            if not seg:
                continue
            if GREEK.match(seg) or not LATIN.search(seg):
                parts.append(seg)
                continue
            span = _italic_span(seg)
            if span is None:
                parts.append(seg)
                continue
            a, b = span
            parts.append(f"{seg[:a]}<i>{seg[a:b]}</i>{seg[b:]}")
        out.append("".join(parts))
    return "\n".join(out)


def lehed():
    """[(nr, pildifail, transkriptsioon)] Lexiconi lehed, järjestatud."""
    rows = []
    for r in csv.DictReader(open(DATA_CSV)):
        name = r["failinimi"].split("/")[-1]
        m = PAGE_NR.match(name)
        if m:
            rows.append((int(m.group(1)), name, r["transkriptsioon"]))
    return sorted(rows)


def kirjuta(out_dir: Path, markup: bool, pildid: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    lk = lehed()
    n_markup = 0
    for nr, pilt, tekst in lk:
        base = f"gezelius-lexicon-{nr:04d}"
        tekst = fix_long_s(fix_ligatures(fix_homoglyphs(tekst)))
        read = tekst.split("\n")
        if read and _is_running_head(read[0]):
            read[0] = fix_running_head(read[0])
            tekst = "\n".join(read)
        if markup and PSEUDO_RANGE[0] <= nr <= PSEUDO_RANGE[1]:
            tekst = pseudo_markup(tekst)
            n_markup += 1
        (out_dir / f"{base}.txt").write_text(tekst.rstrip() + "\n", encoding="utf-8")
        if pildid:
            shutil.copy2(IMAGES / pilt, out_dir / f"{base}.jpg")
            (out_dir / f"{base}.json").write_text(
                json.dumps({"sequence": nr, "status": "Toores"}, indent=2) + "\n",
                encoding="utf-8")
    if pildid:
        (out_dir / "_metadata.json").write_text(json.dumps({
            "slug": "gezelius-lexicon",
            "type": {"id": "Q1261026", "label": "trükis", "source": "wikidata"},
            "tags": [{"id": "Q9129", "label": "kreeka keel", "source": "wikidata"}],
            "title": "Georgius Gezelius, Lexicon Graeco-Latinum",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(lk), n_markup


def stats():
    lk = lehed()
    gr = mixed = 0
    for _, _, t in lk:
        for w in WORD.findall(t):
            if GREEK.search(w):
                gr += 1
                if any(c.isascii() and c.isalpha() for c in w):
                    mixed += 1
    puhas = [fix_long_s(fix_ligatures(fix_homoglyphs(t))) for _, _, t in lk]
    spans = sum(len(re.findall(r"<i>", pseudo_markup(t))) for t in puhas)
    lig = sum(t.count("æ") + t.count("œ") + t.count("Æ") + t.count("Œ")
              for t in puhas)
    longs = sum(t.count("ſ") for t in puhas)
    print(f"Lexiconi lehti:              {len(lk)}")
    print(f"kreekat sisaldavaid sõnu:    {gr}")
    print(f"  homoglüüfidega:            {mixed}  ({100*mixed/gr:.1f} %)")
    print(f"pseudomärgendus annaks:      {spans} <i> spani ({spans/len(lk):.0f} lehe kohta)")
    print(f"ligatuure (æ/œ):             {lig}")
    print(f"pikka s-i (ſ):               {longs}")


def valideeri(kasitsi: Path, pseudo: Path):
    """Võrdleb käsitsi märgendatud lehti reegli väljundiga."""
    failid = sorted(kasitsi.glob("*.txt"))
    if not failid:
        print(f"Viga: {kasitsi} ei sisalda .txt faile."); return
    I = re.compile(r"<i>(.*?)</i>", re.S)
    kokku_k = kokku_p = kattuv = 0
    for f in failid:
        p = pseudo / f.name
        if not p.exists():
            print(f"  ! puudub reegli väljundis: {f.name}"); continue
        k = [re.sub(r"\s+", " ", x).strip() for x in I.findall(f.read_text())]
        r = [re.sub(r"\s+", " ", x).strip() for x in I.findall(p.read_text())]
        ühised = len(set(k) & set(r))
        kokku_k += len(k); kokku_p += len(r); kattuv += ühised
        print(f"  {f.name}: käsitsi {len(k):3d}, reegel {len(r):3d}, kattub {ühised:3d}")
    if kokku_k:
        print(f"\n  recall  (reegel leidis käsitsi omadest): {100*kattuv/kokku_k:.0f} %")
    if kokku_p:
        print(f"  täpsus  (reegli omadest õiged):          {100*kattuv/kokku_p:.0f} %")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--stats" in a:
        stats()
    elif "--export" in a:
        d = Path(a[a.index("--export") + 1])
        n, _ = kirjuta(d, markup=False, pildid=True)
        print(f"Kirjutatud {n} lehte (pilt + txt + json) → {d}")
        print("Tekst on homoglüüfiparandusega, ILMA pseudomärgenduseta.")
    elif "--pseudo" in a:
        d = Path(a[a.index("--pseudo") + 1])
        n, nm = kirjuta(d, markup=True, pildid=False)
        print(f"Kirjutatud {n} .txt → {d}, neist pseudomärgendatud {nm} "
              f"(lk {PSEUDO_RANGE[0]}–{PSEUDO_RANGE[1]})")
        print("SÜNTEETILINE ground truth – valideeri enne kasutamist.")
    elif "--valideeri" in a:
        i = a.index("--valideeri")
        valideeri(Path(a[i + 1]), Path(a[i + 2]))
    else:
        print(__doc__)
