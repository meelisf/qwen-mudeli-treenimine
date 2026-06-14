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
