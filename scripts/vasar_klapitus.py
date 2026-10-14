#!/usr/bin/env python3
"""Vasar 1932 (VUTT tvh4im lk 131–165 = trükk lk 39–73) ↔ jbc88t (põhikiri, Baumanni ärakiri) klapitus (14.10).

Mudeli (v5) read annavad PILDI READ, tekst tuleb Vasarilt (diplomaatilisem: qv, æ, MDXXX). Vasari
teisendused pööratakse tagasi:
  - joonealune „N) Msc.: X" → märgiga sõna(d) asendatakse X-ga (käsikirja lugem); rida saab lipu;
  - toimetaja [lisand] → välja; sõnasisene [laiendus] („J[uris]c[onsul]ti", „v[ero]") → välja + lipp;
  - muu joonealune → märk välja.
Joondus: Riegeri klapituse joonda()/lev() (rieger/klapitus/joonda.py), lehe kaupa Vasari tekstis edasi liikudes.
Väljund: data/vasar/klapitus.json + klapitus.md (rea kaupa, lipud). VUTT-i EI kirjuta.
  venv/bin/python scripts/vasar_klapitus.py   (eeldab scripts/vasar_ocr_jbc88t.py väljundit)
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, "data/rieger/klapitus")
from joonda import joonda, lev, vrd, POOLITUS   # noqa: E402

V = Path("data/vasar")
JRK = "001 w99i7y euc1mm 003 004 005 006 nijmxt 03xao1 ao1wzt w7fej1 vcfr18 7u2ab3 q7fouq 8wrfj2 cnu9tl x52si6 ri7cuq wrzl8o nwyopw xzb6ai w366n3 sx7h30 07sizb 8rs33c 8rmcmn uu36ax t4cecc 65sayd vtb1mb 1cyqbg 25123k 66icu1 7etk59 h5y8f7 wpfud0 5agu0a 021 022".split()
SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
MARK = "\x01"   # lipu ankur Vasari tekstis


def vasari_leht(t):
    read = [re.sub(r"</?m>", "", r).replace("M.ꜰ", "M.te").rstrip() for r in t.split("\n")]
    while read and not read[-1].strip():
        read.pop()
    if read and re.fullmatch(r"\d{1,3}", read[-1].strip()):
        read.pop()                                     # leheküljenumber
    # joonealused algavad ALATI „1)"-ga; põhiteksti punktid on samal kujul („2) Dirigentur", lk 153) →
    # viimane „1)"-rida lehe teises pooles
    k = max((i for i, r in enumerate(read) if re.match(r"\s*1\)\s", r) and i > len(read) // 2), default=len(read))
    keha, jal = read[:k], " ".join(read[k:])
    noot = {int(n): s.strip() for n, s in re.findall(r"(\d{1,2})\)\s*(.*?)(?=\s\d{1,2}\)\s|$)", jal)}
    t = ""
    for r in keha:
        r = r.strip()
        if t.endswith(("-", "¬")) and r[:1].isalpha():
            t = t[:-1] + r
        else:
            t = (t + " " + r) if t else r
    # joonealuse märk: ¹) või ¹ (VUTT OCR); „Msc.: X" → käsikirja lugem
    def asenda(m):
        eel, n = m.group(1), int("".join(SUP[c] for c in m.group(2)))
        s = noot.get(n, "")
        ms = re.match(r"Msc\.:\s*(.+?)\.?$", s)
        if not ms:
            return eel
        lugem = ms.group(1).strip()
        tyhik = eel[len(eel.rstrip()):]              # „geometriam ¹)": märgi ees tühik
        sonad = eel.rstrip().split(" ")
        k = len(lugem.split())
        return (" ".join(sonad[:-k] + [MARK + lugem]) + tyhik) if len(sonad) >= k else eel
    t = re.sub(r"(.*?)([¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s?\)?", asenda, t)
    t = re.sub(r"\s\[[^\]]*\]", " " + MARK, t)              # toimetaja lisand [crebriores]
    t = re.sub(r"\[[^\]]*\]", MARK, t)                       # sõnasisene laiendus J[uris]
    return re.sub(r" +", " ", t)


def _ng(t):
    t = re.sub(r"[^a-z]", "", t.lower().replace("æ", "ae").replace("qv", "qu").replace("&", "et"))
    return {t[i:i + 5] for i in range(len(t) - 4)}


def ankur(P, R, pos):
    g = _ng(P); w = len(P); parim = (-1, pos)
    for x in range(0, max(1, len(R) - w // 2), 150):
        sk = len(g & _ng(R[x:x + w])) - (0 if abs(x - pos) < 600 else 1)
        if sk > parim[0]:
            parim = (sk, x)
    return parim[1]


def main():
    lehed = sorted(V.glob("tvh4im/*.txt"), key=lambda p: int(p.stem[-3:]))
    R = " ".join(vasari_leht(p.read_text(encoding="utf-8")) for p in lehed if 131 <= int(p.stem[-3:]) <= 165)
    lipp_pos = [i for i, c in enumerate(R) if c == MARK]
    R = R.replace(MARK, "")
    lipp_pos = [p - k for k, p in enumerate(lipp_pos)]
    pos, tulem = 0, {}
    for nr, tyvi in enumerate(JRK, 1):
        f = next(V.glob(f"ocr_v5/*-{tyvi}.txt"))
        read = f.read_text(encoding="utf-8").rstrip("\n").split("\n")
        P = "\n".join(read)
        if len(P) < 80:
            tulem[nr] = dict(tyvi=tyvi, read=[(r, None, None, ["lühike"]) for r in read]); continue
        # ankur: käsikirja ja Vasari järjekord võib erineda (lk 27: Caput XVII/XVIII) → parim aken kogu
        # Vasari tekstis märgi-5-grammide kattuvuse järgi; järjestikune positsioon ainult võrdsel korral
        ank = ankur(P, R, pos)
        alg = max(0, ank - 300); lp = min(len(R), ank + int(len(P) * 1.4) + 800)
        kaart, rs, re_ = joonda(P, R[alg:lp])
        piirid, q = [0], 0
        for rida in read[:-1]:
            q += len(rida) + 1; piirid.append(q)
        loiked = [kaart[x] for x in piirid] + [re_]
        Rw = R[alg:lp]
        for li in range(1, len(read)):
            if read[li - 1].rstrip()[-1:] in POOLITUS:
                continue
            b = loiked[li]
            # sidekriipsuta poolitus („fa / cultatum", lk 5): mudelirea viimane sõnaosa lõpeb täpselt lõikekohas
            # sõna SEES → lõiget ei nihutata tühikuni
            viimane = re.sub(r"[^\w]", "", read[li - 1].split()[-1]) if read[li - 1].split() else ""
            if (viimane and 0 < b < len(Rw) and Rw[b].isalpha()
                    and "".join(map(vrd, Rw[max(0, b - len(viimane)):b])) == "".join(map(vrd, viimane))):
                continue
            for d in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                x = b + d
                if 0 < x <= len(Rw) and (x == len(Rw) or Rw[x - 1] == " " or Rw[x] == " "):
                    loiked[li] = x; break
        rr = []
        for li, rida in enumerate(read):
            a, b = alg + loiked[li], alg + loiked[li + 1]
            seg = R[a:b].strip()
            cer = lev(rida.strip(), seg) / max(len(rida.strip()), 1)
            lipud = []
            # Vasari teisenduse tagasipööre (Msc.-lugem, [laiendus] välja) on kinnitatud, kui rida klapib
            # mudeliga (14.10: 148/215 cer ≤ 0,10) → lipp ainult lahknemisel
            if any(a <= p < b for p in lipp_pos) and cer > 0.10:
                lipud.append("vasar_teisendus")
            if cer > 0.35:
                lipud.append(f"cer{cer:.2f}")
            rr.append((rida, seg, round(cer, 3), lipud))
        cs = [c for _, _, c, _ in rr]
        if sorted(cs)[len(cs) // 2] > 0.4:              # vasteta leht (tiitel, register): positsioon jääb
            tulem[nr] = dict(tyvi=tyvi, vasteta=True, read=[(r, None, None, ["vasteta"]) for r in read])
            print(f"lk {nr:2d} {tyvi} VASTETA (mediaan cer {sorted(cs)[len(cs) // 2]:.2f})", flush=True)
            continue
        tulem[nr] = dict(tyvi=tyvi, alg=alg + rs, lopp=alg + re_, read=rr)
        pos = alg + re_
        print(f"lk {nr:2d} {tyvi} R[{alg + rs}:{alg + re_}]/{len(R)} read {len(rr):2d} kesk {sum(cs)/len(cs):.2f} "
              f"max {max(cs):.2f} lipuga {sum(1 for x in rr if x[3])}", flush=True)
    json.dump(tulem, open(V / "klapitus.json", "w"), ensure_ascii=False, indent=1)
    with open(V / "klapitus.md", "w", encoding="utf-8") as g:
        for nr, t in tulem.items():
            g.write(f"\n## lk {nr} ({t['tyvi']}) https://vutt.utlib.ut.ee/work/jbc88t/{nr}\n\n")
            for rida, seg, cer, lip in t["read"]:
                g.write(f"- {'⚑ ' + ','.join(lip) + ' ' if lip else ''}`{rida}`\n  → `{seg}`\n")


if __name__ == "__main__":
    main()
