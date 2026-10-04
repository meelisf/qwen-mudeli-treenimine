#!/usr/bin/env python3
"""Lühendusmärk → makron (VUTT ADR 0062, issue #533).

Reegel: rõhtjoon tähe kohal (lühend, geminatsioonikriips) on U+0304 COMBINING
MACRON. Teisendatakse ainult LADINA tähe kohal:
  U+0303 tilde        → U+0304   (DTA m̃ ñ, ẽ, ã …; precomposed lagundatakse NFD-ga)
  U+0305 ülakriips    → U+0304   (xix 1800–49)
  topelt U+0304       → üks
Puutumata: kreeka (U+0342 perispomeni ja tilde kreeka tähel), numbrid (vinculum),
eraldiseisev „~" (kordusmärk). Väljund NFC (ā ē ī ō ū precomposed, m̄ n̄ q̄ kombineeriv).

Keelevalvur (est/spa/por, ADR 0062 p 3) ei ole siin vajalik: Kurrendi allikates
neid keeli ei ole. VUTT-i korpuse teisendus kasutab sama kaarti koos valvuriga —
kaart PEAB kahes kohas kattuma (test: `python3 scripts/lyhend_makron.py --test`).
"""
import sys
import unicodedata as ud

TILDE, MAKRON, YLAKRIIPS = "̃", "̄", "̅"


def _ladina_taht(ch):
    return ch.isalpha() and "LATIN" in ud.name(ch, "")


def makroniks(text):
    """Tagastab (uus_tekst, muudetud_märkide_arv)."""
    s = ud.normalize("NFD", text)
    out, muudetud, alus = [], 0, ""
    for ch in s:
        if ud.combining(ch) == 0:
            alus = ch
            out.append(ch)
            continue
        if ch in (TILDE, YLAKRIIPS) and _ladina_taht(alus):
            ch = MAKRON
            muudetud += 1
        if ch == MAKRON and out and out[-1] == MAKRON:
            muudetud += 1                          # topeltmakron → üks
            continue
        out.append(ch)
    return ud.normalize("NFC", "".join(out)), muudetud


def _test():
    juhud = [
        ("Camm̃erherr", "Camm̄erherr"),
        ("cũ nõ dẽ", "cū nō dē"),
        ("vñ weñ", "vn̄ wen̄"),
        ("q̃", "q̄"),
        ("m̅", "m̄"),
        ("8̅", "8̅"),               # vinculum numbril jääb
        ("ῖ υ̃", "ῖ υ̃"),           # kreeka jääb
        ("a ~ b", "a ~ b"),                   # kordusmärk jääb
        ("ā̄", "ā"),                     # topelt → üks
        ("Õ", "Ō"),
    ]
    for sisend, oodatud in juhud:
        tul, _ = makroniks(sisend)
        assert tul == ud.normalize("NFC", oodatud), (sisend, tul, oodatud)
    print(f"ok ({len(juhud)} juhtu)")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
