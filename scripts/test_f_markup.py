"""`<f>` kanoniseerimise, poolitusvärava ja mõõdiku testid (VUTT ADR 0077).

Käivitus: `venv/bin/python -m pytest scripts/test_f_markup.py` või
`venv/bin/python scripts/test_f_markup.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_marginalia import clean_markup  # noqa: E402
from f_markup import (  # noqa: E402
    canonicalize_page_mark_order, canonicalize_style_tags, cs_to_f, f_f1, f_letter_share, hyphen_gate_violations,
)


def test_servatuhik_valja():
    assert canonicalize_style_tags("<f>den </f>exercitiis<f> finden</f>") == \
        "<f>den</f> exercitiis <f>finden</f>"


def test_korvuti_paarid_kokku():
    assert canonicalize_style_tags("<f>A</f> <f>B</f>") == "<f>A B</f>"
    assert canonicalize_style_tags("<i>a</i><i>b</i>") == "<i>ab</i>"
    assert canonicalize_style_tags("<f>A</f> <i>B</i>") == "<f>A</f> <i>B</i>"


def test_clean_markup_kanoniseerib():
    # <m> välimiseks tõstab VUTT salvestamisel (normalize_marginalia_tags), mitte siin
    assert clean_markup("<m><f>Ge⸗ </f></m>") == "<m><f>Ge⸗</f></m>"
    assert clean_markup("<f>den </f>x<f> und</f>") == "<f>den</f> x <f>und</f>"


def test_poolitusvarav():
    assert hyphen_gate_violations("<f>vnd Ge⸗</f>\n<f>witter</f>") == []
    assert hyphen_gate_violations("<m><f>Ge⸗</f></m>") == []
    assert hyphen_gate_violations("vnd Ge⸗\nwitter") == [1]          # ⸗ väljas
    assert hyphen_gate_violations("<f>ex-</f>\nemplum") == [1]        # - sees
    assert hyphen_gate_violations("anno <f>1640-</f>") == []          # vahemik, mitte poolitus
    assert hyphen_gate_violations("ex-\nemplum <f>Gott</f>") == []


def test_cs_to_f_peegel():
    assert cs_to_f("Thesis <cs>Gott</cs> prima", "A") == "Thesis <f>Gott</f> prima"
    assert cs_to_f("sich bey den <cs>exercitiis</cs> finden", "F") == \
        "<f>sich bey den</f> exercitiis <f>finden</f>"


def test_toimetaja_marge_jaab_valja():
    assert cs_to_f("[tühi lehekülg]", "F") == "[tühi lehekülg]"


def test_kustos_enne_signatuuri():
    assert canonicalize_page_mark_order("tekst\n<sig>C3</sig>\n<catch>Mul¬</catch>") == \
        "tekst\n<catch>Mul¬</catch>\n<sig>C3</sig>"
    assert canonicalize_page_mark_order("tekst\n<sig>B</sig> <catch>repe-</catch>\n") == \
        "tekst\n<catch>repe-</catch>\n<sig>B</sig>\n"
    ok = "tekst\n<catch>Vor⸗</catch>\n<sig>A ij</sig>"
    assert canonicalize_page_mark_order(ok) == ok
    assert canonicalize_page_mark_order("a <sig>B</sig>\n<catch>x</catch>") == "a <sig>B</sig>\n<catch>x</catch>"
    # ühel real (04na2fu4 lk 49)
    assert canonicalize_page_mark_order("tekst\n<sig>G</sig>  <catch>Mittel</catch>") == \
        "tekst\n<catch>Mittel</catch>\n<sig>G</sig>"
    # topeltlehekülg: iga <pb/> osa eraldi; tägirida (<i>) enne <pb/>-d jäetakse vahele
    t = "a\n<sig>B</sig>\n<catch>Vor</catch>\n<i>\n<pb/>\nb</i>\n<sig>C</sig>\n<catch>x</catch>"
    assert canonicalize_page_mark_order(t) == \
        "a\n<catch>Vor</catch>\n<sig>B</sig>\n<i>\n<pb/>\nb</i>\n<catch>x</catch>\n<sig>C</sig>"


def test_juhised():
    from prompt import INSTRUCTION, INSTRUCTION_CS
    assert "<f>text</f>" in INSTRUCTION and "<cs>" not in INSTRUCTION
    assert "<sig>A ij</sig>" in INSTRUCTION and "<catch>" in INSTRUCTION
    assert "as the LAST line" in INSTRUCTION
    # OCR-teenuse juhis jääb vanaks kuni mudeli vahetuseni
    assert "<cs>text</cs>" in INSTRUCTION_CS and "<sig>" not in INSTRUCTION_CS


def test_moodik():
    assert f_f1("a <f>Gott</f>", "a <f>Gott</f>") == 1.0
    assert f_f1("a b", "a b") == 1.0
    assert f_f1("<f>Gott</f>", "Gott") == 0.0
    assert round(f_letter_share("<f>abcd</f> ef"), 2) == 0.67


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
