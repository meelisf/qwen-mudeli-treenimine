#!/usr/bin/env python3
"""Vasari klapitus → VUTT-i kirjutatav lehetekst (jbc88t, 15.10).

Sisend data/vasar/klapitus.json (scripts/vasar_klapitus.py). Rea tekst:
  - Vasari lõik, kui rea CER ≤ 0,35; muidu mudeli (v5) rida — lipp kommentaari;
  - rea lõpu poolitus (mudeli ¬) → - (käsikirjas ühekordne, kasutaja 15.10);
  - käsikirja „&": Vasar ja v5 kirjutavad enamasti „et"; kui mudel luges samas kohas „&", jääb „&".
Vasteta / lühikesed lehed (tiitel, märkmed, tühi lk) jäetakse välja — nende VUTT-i tekst jääb puutumata.
Väljund data/vasar/vasar_vutti.json: [{leht, nr, mudel, tekst, lipud: [{rida, pohjus, lehel, teine}]}].
  venv/bin/python scripts/vasar_vutti.py
"""
import difflib, json, re
from pathlib import Path

V = Path("data/vasar")
TAU = 0.35


def taasta_ja(ocr, seg):
    """Mudeli „&" ↔ Vasari „et" samal kohal → „&" (sõnajoondus)."""
    a, b = ocr.split(), seg.split()
    if "&" not in "".join(a):
        return seg
    sm = difflib.SequenceMatcher(None, [w.lower().strip(",.;:") for w in a], [w.lower().strip(",.;:") for w in b],
                                 autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if a[i].startswith("&") and re.fullmatch(r"et[,.;:]?", b[j]):
                    b[j] = "&" + b[j][2:]
    return " ".join(b)


def main():
    d = json.load(open(V / "klapitus.json", encoding="utf-8"))
    out = []
    for nr, t in d.items():
        if t.get("vasteta") or all("lühike" in r[3] for r in t["read"]):
            continue
        read, lipud = [], []
        for i, (ocr, seg, cer, lip) in enumerate(t["read"], 1):
            poolitus = ocr.rstrip()[-1:] in "¬⸗=-"
            if seg is not None and cer <= TAU:
                tekst = re.sub(r" {2,}", " ", taasta_ja(ocr, seg)).rstrip().rstrip("¬⸗=-")   # „vel [in] vita"
                if poolitus:
                    tekst += "-"
                if lip:
                    lipud.append(dict(rida=i, pohjus="Vasari joonealune (Msc.) / laiendus, mudel luges teisiti",
                                      lehel="Vasar", teine=ocr))
            else:
                tekst = re.sub(r"[¬=⸗]$", "-", ocr.rstrip())
                lipud.append(dict(rida=i, pohjus=f"rida ei klapi (CER {cer:.2f})" if seg else "vasteta",
                                  lehel="OCR", teine=seg or ""))
            read.append(tekst)
        out.append(dict(leht=t["tyvi"], nr=int(nr), mudel="\n".join(r[0] for r in t["read"]),
                        tekst="\n".join(read), lipud=lipud))
    json.dump(out, open(V / "vasar_vutti.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"lehti {len(out)}, ridu {sum(o['tekst'].count(chr(10)) + 1 for o in out)}, "
          f"lipuga {sum(len(o['lipud']) for o in out)}, &-ga {sum(o['tekst'].count('&') for o in out)}, "
          f"lehti lipuga {sum(1 for o in out if o['lipud'])}")


if __name__ == "__main__":
    main()
