"""Absoluutne fraktuurimärgend `<f>` treeningandmetes (VUTT ADR 0077).

VUTT märgib fraktuuri alates 2026-11 ALATI `<f>`-iga; märgendamata ladina tähestik
on antiikva. Varem oli `<cs>` suhteline (teose põhikirja suhtes) ja sama pilt sai
teose kaupa vastupidise sihi.

Siin:
- `canonicalize_style_tags`: servatühik välja, kõrvuti paarid kokku (`<i>`, `<b>`, `<f>`)
  — kirja pildil nähtamatud erinevused, mis annaksid mudelile kõikuva sihi
  (ADR 0077 p 5). Kutsub `clean_markup`.
- `hyphen_gate_violations`: rea lõpu `⸗` peab olema `<f>`-i sees ja `-` väljas
  (ADR 0077 p 4). Rikkumisega leht jääb ekspordist välja, kui ta pole erandite
  nimekirjas (`data/f_poolitus_erandid.txt`).
- `cs_to_f`: VUTT `server/script_migration.cs_to_f` peegel — vana mudeli suhtelise
  väljundi teisendus hindamisel (aus võrdlus uue mudeliga).
- `f_spans`, `f_f1`: `<f>`-vahemike mõõdik hindamiseks.
"""
import re
import unicodedata
from collections import Counter

STYLE_TAGS = ("i", "b", "f")
_TOKEN_RE = re.compile(r"<(/?)([a-z]+)(\d*)(/?)>")
_TRAILING_CLOSE_RE = re.compile(r"(?:</[a-z]+\d*>|\s)+$")
_ALT = "|".join(STYLE_TAGS)
_OPEN_WS_RE = re.compile(rf"<({_ALT})>([ \t]+)")
_CLOSE_WS_RE = re.compile(rf"([ \t]+)</({_ALT})>")
_ADJ_RE = re.compile(rf"</({_ALT})>([ \t]*)<\1>")


def canonicalize_style_tags(text: str) -> str:
    """`<f>den </f>` → `<f>den</f> `; `<f>A</f> <f>B</f>` → `<f>A B</f>`. Idempotentne."""
    prev = None
    while prev != text:
        prev = text
        text = _OPEN_WS_RE.sub(r"\2<\1>", text)
        text = _CLOSE_WS_RE.sub(r"</\2>\1", text)
        text = _ADJ_RE.sub(r"\2", text)
    return text


def hyphen_gate_violations(text: str) -> list:
    """Reanumbrid (1-põhised), kus rea lõpu poolitusmärk ja `<f>` ei klapi.

    Rea viimane nähtav märk enne sulgevaid täge; `-` ainult tähe järel (`1640-` ei
    ole poolitus). `<f>` olek on rea lõpus, enne sulgevaid täge."""
    bad = []
    f_open = 0
    for no, line in enumerate((text or "").split("\n"), 1):
        core = _TRAILING_CLOSE_RE.sub("", line)
        for m in _TOKEN_RE.finditer(core):
            if m.group(2) == "f" and not m.group(4):
                f_open = max(0, f_open + (-1 if m.group(1) else 1))
        if core.endswith("⸗") and not f_open:
            bad.append(no)
        elif len(core) >= 2 and core.endswith("-") and core[-2].isalpha() and f_open:
            bad.append(no)
        for m in _TOKEN_RE.finditer(line[len(core):]):
            if m.group(2) == "f" and not m.group(4):
                f_open = max(0, f_open + (-1 if m.group(1) else 1))
    return bad


def load_exceptions(path) -> set:
    """Kontrollitud erandid: üks lehe võti (`teos/lehe_tüvi`) rea kohta, `#` kommentaar.
    Nt trükikoda on fraktuuris päriselt kasutanud ühekordset `-`."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return set()
    return {ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()}


# --- VUTT `cs_to_f` peegel (server/script_migration.py) ---

def _is_latin(ch: str) -> bool:
    return ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN")


_EDITORIAL_RE = re.compile(r"\[[^\]\n]*\]")


def _wrap_runs(segment: str) -> str:
    """Toimetaja märge nurksulgudes (`[tühi lehekülg]`, `[vasakpoolne lk]`) ei ole trükitekst
    ega lähe `<f>`-i — tühja lehe märgend peab jääma baidi-täpseks."""
    out, pos = [], 0
    for m in _EDITORIAL_RE.finditer(segment):
        out.append(_wrap_runs_plain(segment[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_wrap_runs_plain(segment[pos:]))
    return "".join(out)


def _wrap_runs_plain(segment: str) -> str:
    out, piece = [], []

    def flush():
        if not piece:
            return
        s = "".join(piece)
        core = s.strip()
        if core and any(_is_latin(c) for c in core):
            lead = s[: len(s) - len(s.lstrip())]
            trail = s[len(s.rstrip()):]
            out.append(f"{lead}<f>{core}</f>{trail}")
        else:
            out.append(s)
        piece.clear()

    for ch in segment:
        if ch.isalpha() and not _is_latin(ch):
            flush()
            out.append(ch)
        else:
            piece.append(ch)
    flush()
    return "".join(out)


def cs_to_f(text: str, base: str) -> str:
    """Suhteline `<cs>` → absoluutne `<f>`; base 'A' | 'F' (VUTT-iga identne)."""
    if base not in ("A", "F"):
        raise ValueError(base)
    if "<f>" in text and "<cs>" not in text:
        return text
    open_counts = {"cs": 0, "i": 0, "fn": 0}
    out, pos = [], 0

    def emit(seg):
        if not seg:
            return
        fraktur = (base == "F") != bool(open_counts["cs"])
        if open_counts["fn"] or open_counts["i"] or not fraktur:
            out.append(seg)
        else:
            out.append("\n".join(_wrap_runs(ln) for ln in seg.split("\n")))

    for m in _TOKEN_RE.finditer(text):
        emit(text[pos:m.start()])
        pos = m.end()
        name = m.group(2)
        if name in open_counts and not m.group(4):
            open_counts[name] = max(0, open_counts[name] + (-1 if m.group(1) else 1))
        if name != "cs":
            out.append(m.group(0))
    emit(text[pos:])
    return "".join(out)


def f_letter_share(text: str) -> float:
    """Ladina tähtede osa `<f>`-is (lehe põhikiri hindamisel)."""
    f = a = 0
    depth = 0
    pos = 0
    for m in list(_TOKEN_RE.finditer(text)) + [None]:
        seg = text[pos:m.start()] if m else text[pos:]
        n = sum(1 for c in seg if _is_latin(c))
        if depth:
            f += n
        else:
            a += n
        if m is None:
            break
        pos = m.end()
        if m.group(2) == "f" and not m.group(4):
            depth = max(0, depth + (-1 if m.group(1) else 1))
    return f / (f + a) if f + a else 0.0


# --- mõõdik ---

def f_spans(text: str) -> Counter:
    """`<f>`-vahemike sisu (tühikud ühtlustatud, tägid välja) multihulgana."""
    spans = Counter()
    for m in re.finditer(r"<f>(.*?)</f>", text or "", re.S):
        s = " ".join(re.sub(r"<[^>]+>", "", m.group(1)).split())
        if s:
            spans[s] += 1
    return spans


def f_f1(ref: str, hyp: str) -> float:
    """`<f>`-vahemike F1 (täpne vaste). Mõlemas null vahemikku → 1.0."""
    r, h = f_spans(ref), f_spans(hyp)
    if not r and not h:
        return 1.0
    tp = sum((r & h).values())
    p = tp / max(sum(h.values()), 1)
    rc = tp / max(sum(r.values()), 1)
    return 2 * p * rc / (p + rc) if p + rc else 0.0
