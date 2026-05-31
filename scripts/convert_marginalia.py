#!/usr/bin/env python3
"""
Konverteerib inline <m>...</m> ääremärkused ankrutega formaati.

Inline formaat (praegune):
    ...Circe,
    <m>Chryſoſt.</m>
    <m>tom: 3. in</m>
    ...
    Medea: Im gleichen...

Ankrutega formaat (uus):
    ...Circe, <m_ref id="1"/>
    Medea: Im gleichen...

    [MARGINAALID]
    <m id="1">Chryſoſt.
    tom: 3. in
    ...</m>

Käivitamine üksikfailil:
    python scripts/convert_marginalia.py path/to/page.txt

Partii-režiim (prindi ainult):
    python scripts/convert_marginalia.py --test path/to/page.txt
"""

import re
import sys
from pathlib import Path


# Kas kogu rida on üks <m>...</m> blokk (standalone marginaalia rida)
_STANDALONE_M = re.compile(r"^<m>(.*)</m>$", re.DOTALL)

# Inline <m>...</m> segatud reas (mitu tagi või tekst ümber)
_INLINE_M = re.compile(r"<m>(.*?)</m>", re.DOTALL)


def convert(text: str) -> str:
    """Teisendab inline marginaaliad ankrutega formaati.

    Tagastab muutmata teksti kui marginaalid puuduvad.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    marginalia_blocks: list[list[str]] = []  # iga blokk = ridade list
    current_block: list[str] = []

    def _flush_block() -> None:
        """Sulgeb jooksva bloki ja lisab ankru eelmisele põhiteksti reale."""
        if not current_block:
            return
        block_id = len(marginalia_blocks) + 1
        marginalia_blocks.append(current_block[:])
        current_block.clear()
        if result_lines:
            result_lines[-1] = result_lines[-1].rstrip() + f' <m_ref id="{block_id}"/>'

    for line in lines:
        stripped = line.strip()
        m = _STANDALONE_M.match(stripped)

        if m:
            # Kogu rida on marginaalia sisu
            current_block.append(m.group(1))
        else:
            # Põhiteksti rida — sulge eelmine blokk (kui oli)
            _flush_block()

            # Käsitle inline <m>...</m> segatud reas
            inline_ids: list[int] = []

            def _replace_inline(match: re.Match) -> str:
                block_id = len(marginalia_blocks) + len(inline_ids) + 1
                inline_ids.append(block_id)
                marginalia_blocks.append([match.group(1)])
                return f'<m_ref id="{block_id}"/>'

            processed = _INLINE_M.sub(_replace_inline, line)
            result_lines.append(processed)

    # Sulge viimane blokk (kui tekst lõpeb marginaaliaga)
    _flush_block()

    if not marginalia_blocks:
        return text

    # Ehita [MARGINAALID] sektsioon
    m_parts = []
    for i, block_lines in enumerate(marginalia_blocks):
        content = "\n".join(block_lines)
        m_parts.append(f'<m id="{i + 1}">{content}</m>')

    marginalia_section = "[MARGINAALID]\n" + "\n\n".join(m_parts)
    main_text = "\n".join(result_lines).rstrip()

    return f"{main_text}\n\n{marginalia_section}"


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

        has_m = bool(_STANDALONE_M.search(original) or _INLINE_M.search(original))
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
            blocks = converted.count("<m id=")
            print(f"{path}: {blocks} marginaalia blokki konverteeritud")


if __name__ == "__main__":
    main()
