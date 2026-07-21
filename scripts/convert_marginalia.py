#!/usr/bin/env python3
"""
Liigutab inline <m>...</m> ääremärkused lehe lõppu.

Inline formaat (sisend):
    ...Circe,
    <m>Chryſoſt.</m>
    <m>tom: 3. in</m>
    ...
    Medea: Im gleichen...

Väljundformaat:
    ...Circe,
    ...
    Medea: Im gleichen...
    <m>Chryſoſt.
    tom: 3. in
    ...</m>

Topeltleheküljel (<pb/>) pannakse <m> plokid iga lehe osa lõppu:
    vasak tekst...
    <m>vasaku lehe märkus</m>
    <pb/>
    parem tekst...
    <m>parema lehe märkus</m>

Käivitamine üksikfailil:
    python scripts/convert_marginalia.py path/to/page.txt

Partii-režiim (prindi ainult):
    python scripts/convert_marginalia.py --test path/to/page.txt
"""

import re
import sys
from pathlib import Path


# Mitmerealisel <m>...</m> plokil on sisu üle mitme rea — eeltöötlus jagab need ühereallisteks.
_MULTILINE_M = re.compile(r"<m>(.*?)</m>", re.DOTALL)


def _normalize_multiline_m(text: str) -> str:
    """Jagab mitmerealised <m>...</m> plokid järjestikusteks ühereallisteks <m> kirjeteks."""
    def _split(match: re.Match) -> str:
        content = match.group(1)
        if "\n" not in content:
            return match.group(0)
        stripped = content.strip()
        fmt_m = re.match(r"^<([ib])>(.*)</\1>$", stripped, re.DOTALL)
        if fmt_m:
            tag, inner = fmt_m.group(1), fmt_m.group(2)
            return "\n".join(f"<m><{tag}>{line}</{tag}></m>" for line in inner.split("\n"))
        return "\n".join(f"<m>{line}</m>" for line in content.split("\n"))

    return _MULTILINE_M.sub(_split, text)


# Tuvastab kõik eraldi-rea marginaalide variandid
_M_LINE_RE = re.compile(
    r"^"
    r"(?:<cs>)?"
    r"(?P<outer_fmt><[ib]>)?"
    r"<m>"
    r"(?P<inner_fmt><[ib]>)?"
    r"(?P<content>.*?)"
    r"(?:</[ib]>)?"
    r"</m>"
    r"(?:</[ib]>)?"
    r"(?:</cs>)?"
    r"$",
    re.DOTALL,
)

# Inline <m>...</m> segatud reas (tekst ümber)
_INLINE_M = re.compile(r"<m>(.*?)</m>", re.DOTALL)


def _extract_m_content(line: str) -> str | None:
    """Tagastab marginaalia sisu stringina, või None kui rida pole marginaalia."""
    m = _M_LINE_RE.match(line.strip())
    if m is None:
        return None
    content = m.group("content")
    fmt = m.group("outer_fmt") or m.group("inner_fmt")
    if fmt:
        tag = fmt[1:-1]
        content = f"<{tag}>{content}</{tag}>"
    return content


def convert(text: str) -> str:
    """Liigutab kõik <m>sisu</m> plokid lehe lõppu.

    <pb/> piiri arvestatakse: iga lehe osa saab oma <m> plokid vahetult oma teksti järele.
    """
    text = _normalize_multiline_m(text)

    if not (_M_LINE_RE.search(text) or _INLINE_M.search(text)):
        return text

    # Jagame <pb/> piiri järgi, säilitades eraldaja
    parts = re.split(r"(<pb/>)", text)
    result_parts = []
    any_changed = False

    for part in parts:
        if part == "<pb/>":
            result_parts.append(part)
            continue

        lines = part.split("\n")
        main_lines = []
        m_blocks = []
        current_m_lines: list[str] = []

        for line in lines:
            m_content = _extract_m_content(line)
            if m_content is not None:
                current_m_lines.append(m_content)
            else:
                if current_m_lines:
                    m_blocks.append("\n".join(current_m_lines))
                    current_m_lines = []

                inline_found: list[str] = []
                def _repl(match: re.Match, _buf: list = inline_found) -> str:
                    _buf.append(match.group(1))
                    return ""
                processed = _INLINE_M.sub(_repl, line)
                if inline_found:
                    m_blocks.extend(inline_found)
                    processed = re.sub(r"  +", " ", processed).strip()
                main_lines.append(processed)

        if current_m_lines:
            m_blocks.append("\n".join(current_m_lines))

        if m_blocks:
            any_changed = True
            section_text = "\n".join(main_lines).rstrip()
            m_text = "\n".join(f"<m>{c}</m>" for c in m_blocks)
            result_parts.append(section_text + "\n" + m_text)
        else:
            result_parts.append(part)

    if not any_changed:
        return text

    return "".join(result_parts)


def remove_empty_m_tags(text: str) -> str:
    """Eemaldab tühjad <m>...</m> tagid (nt <m></m>, <m><i></i></m>)."""
    def _is_empty(content: str) -> bool:
        s = content.strip()
        while s:
            new = re.sub(r"<\w+>\s*</\w+>", "", s).strip()
            if new == s:
                break
            s = new
        return not s

    result = re.sub(
        r"<m>(.*?)</m>",
        lambda m: "" if _is_empty(m.group(1)) else m.group(0),
        text,
        flags=re.DOTALL,
    )
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


#: Tagid, mis VUTT-ist tulevad, aga ei kuulu treeningjuhisesse (prompt.py).
#: Sisu on päris lehetekst (käsikirjalised parandused trükitud lehel), seega
#: eemaldatakse ainult märgend ise, sisu jääb alles. Ühtlasi parandab see
#: katkise pesastuse: just <annN> on see, mis lõikub üle <i>/<cs>/<m> piiride.
UNWRAP_TAGS = ("ann1", "ann2", "ann3", "ann4")

#: <noodid> jääb ALLES – see märgib kohta, kus lehel on noodikiri. Ilma selleta
#: satub mudel noote nähes segadusse ja hakkab neid transkribeerida püüdma;
#: märgendiga paneb ta märke ja liigub edasi.


def unwrap_tags(text: str, tags: tuple = UNWRAP_TAGS) -> str:
    """Eemaldab nimetatud tagid, säilitades nende sisu.

    <ann1>Schmolentzkow</ann1>/ Twertſky/  →  Schmolentzkow/ Twertſky/
    """
    for tag in tags:
        text = re.sub(rf"</?{tag}\s*/?>", "", text)
    return text


#: Märgendid ilma paarilise sulgejata – neid pesastuse kontroll ei arvesta.
#: <pb/> on isesulguv, <noodid> on üksik marker (vt selgitust ülal).
_UNPAIRED = ("pb", "noodid")

_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)\s*(/?)>")


def fix_crossed_tags(text: str) -> str:
    """Parandab ristuva pesastuse, järjestades sulgejad avamisjärjekorda.

    VUTT-i transkriptsioonides on levinud näpukas, kus sisemine ja välimine
    märgend suletakse vales järjekorras – eriti poolitatud sõnades ja
    marginaaliaridades:

        <i>L. Baro in <cs>Ekeby⸗</i></cs>   →  <i>L. Baro in <cs>Ekeby⸗</cs></i>
        <m><i>traria.</m></i>               →  <m><i>traria.</i></m>

    Kui sulgeja vastab mõnele pinus allpool olevale avajale, suletakse kõik
    selle peal olevad märgendid ja avatakse pärast uuesti – nii säilib
    algne kavatsetud ulatus. Tekkivad tühjad paarid koristab
    remove_empty_tags().

    Avajata sulgejaid ja sulgejata avajaid EI puudutata: need on
    leheküljepiiri ületavad jooksud (kaldkiri algab eelmisel lehel), mis on
    lehekaupa transkribeerimisel täiesti õiguspärased.
    """
    out = []
    stack = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        closing, name, selfc = m.group(1), m.group(2).lower(), m.group(3)
        out.append(text[pos:m.start()])
        pos = m.end()

        if name in _UNPAIRED or selfc:
            out.append(m.group(0))
        elif not closing:
            stack.append(name)
            out.append(m.group(0))
        elif stack and stack[-1] == name:
            stack.pop()
            out.append(m.group(0))
        elif name in stack:
            # Ristuv: sulge vahepealsed, sulge see, ava vahepealsed uuesti.
            idx = len(stack) - 1 - stack[::-1].index(name)
            inner = stack[idx + 1:]
            out.append("".join(f"</{x}>" for x in reversed(inner)))
            out.append(f"</{name}>")
            out.append("".join(f"<{x}>" for x in inner))
            del stack[idx]
        else:
            out.append(m.group(0))   # avajata sulgeja – lehepiiri jooks

    out.append(text[pos:])
    return "".join(out)


def remove_empty_tags(text: str) -> str:
    """Eemaldab sisuta märgendipaarid (<i></i>, <cs></cs>, <m><i></i></m> …).

    Korratakse püsipunktini, et ka pesastatud tühjad plokid kaoksid.
    <pb/> ja <noodid> jäävad puutumata – neil pole paarilist.
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"<([a-zA-Z0-9]+)>\s*</\1>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test_mode = "--test" in sys.argv

    if not args:
        print("Kasutus: python scripts/convert_marginalia.py [--test] fail.txt [fail2.txt ...]")
        sys.exit(1)

    for path_str in args:
        path = Path(path_str)
        if not path.exists():
            print(f"Viga: {path} ei leitud")
            continue

        original = path.read_text(encoding="utf-8")
        converted = convert(original)

        has_m = bool(_M_LINE_RE.search(original) or _INLINE_M.search(original))
        changed = converted != original

        if test_mode or not changed:
            print(f"\n{'='*60}")
            print(f"Fail: {path}")
            if not has_m:
                print("(marginaalid puuduvad, muudatusi pole)")
            else:
                print(converted)
        else:
            path.write_text(converted, encoding="utf-8")
            blocks = converted.count("<m>")
            print(f"{path}: {blocks} marginaalia blokki lehe lõppu liigutatud")


if __name__ == "__main__":
    main()
