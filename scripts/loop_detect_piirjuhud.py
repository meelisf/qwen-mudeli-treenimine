#!/usr/bin/env python3
"""Kordusloopi tuvastaja piirjuhud: trükitud märgiread EI OLE loop, päris loop on.

    venv/bin/python scripts/loop_detect_piirjuhud.py    # exit = vigade arv

Vana tuvastaja (enne 11.10.2026) kukkus läbi 4 juhul 8-st — vt `PUNCT_MIN_REPS`.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from loop_detect import is_looped
alus = "Es wäre unser Winkelmann wol auch nicht gewesen, der er ward! "
juhud = {
    "ärajätt . . . lehe lõpus":   (alus + ". . . Seit", False),   # Rieger lk 100
    "lõiguvahe * * * lõpus":      (alus + "\n* * *\nNeue", False),
    "mõttekriipsud — — — — lõpus": (alus + "— — — —", False),
    "punktirida 40 märki lõpus":  (alus + "." * 40, False),
    "päris loop '. ' tokenilaeni": (alus + ". " * 800 + ".", True),
    "päris loop '—' märgid":      (alus + "—" * 600, True),
    "päris loop ææææ":            (alus + "æ" * 300, True),
    "päris sõnaloop":             (alus + "und die Stadt " * 40 + "un", True),
}
vigu = 0
for nimi, (tekst, oodatud) in juhud.items():
    tulem = bool(is_looped(tekst))
    ok = tulem == oodatud
    vigu += not ok
    print("OK  " if ok else "VIGA", nimi, is_looped(tekst))
sys.exit(vigu)
