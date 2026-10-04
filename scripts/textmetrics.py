#!/usr/bin/env python3
"""
Tekstimõõdikud – jagatud loogika.

Need funktsioonid on identsed `eval_print.py` omadega (cer, strip_tags,
normaliseeri, strip_output). `eval_print.py` hoiab praegu oma koopiat ja seda
EI OLE siia peale lülitatud meelega: ta on tootmises kasutuses ja teda ei
tahtnud mõõtmispäeval puutuda. Kui `eval_print.py` järgmine kord niikuinii
muutub, tuleks ta siia peale tõsta – muidu jooksevad kaks koopiat lahku, nagu
juhtus `loop_detect.py` ja teenuse `LoopStopper`-iga (vt SEIS §5.6).
"""
import re

import editdistance

from lyhend_makron import makroniks

TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def strip_tags(text: str) -> str:
    return re.sub(r"[ \t]+", " ", TAG.sub("", text)).strip()


def normaliseeri(s: str) -> str:
    """Sisu võrdlemiseks: märgendid, kirjavahemärgid ja tühik maha.
    Lühendusmärk tilde ≡ makron (VUTT ADR 0062): vana mudel kirjutab tilde, GT makronit."""
    s = makroniks(strip_tags(s))[0].lower()
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE)


def strip_output(text: str) -> str:
    """Sama puhastus mis teenuses: think-plokid, assistendi markerid, koodiplokid."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for marker in ["</assistant>", "<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()
