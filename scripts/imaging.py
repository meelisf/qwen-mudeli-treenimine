#!/usr/bin/env python3
"""
Pildieelarve – jagatud konstant ja skaleerimine.

Qwen3.5 image processor'i `longest_edge` EI ole serva pikkus, vaid KOGU
pikslite arv. Valem: visuaaltokeneid ≈ pikslid / 1024, ehk 5 120 000 px
annab ~5000 tokenit pildi kohta. Vaikeväärtus 16M px tähendab ~14 000
tokenit ja OOM-i.

Konstant elab siin, sest teda vajavad kaks skripti korraga:
  - train_markup.py  – seab image_processor'i eelarve
  - build_vutt_dataset.py – skaleerib pildid juba andmestiku ehitamisel

Kui need kaks lahku jooksevad, on tagajärg vaikne: eelarve tõstmine ei
annaks mingit võitu, sest kettal olevad pildid oleks juba väiksemaks
tehtud. Sellepärast üks allikas.
"""

from pathlib import Path
import shutil

from PIL import Image

#: Maksimaalne pikslite arv pildi kohta (~5000 visuaaltokenit).
MAX_PIXELS = 5_120_000

#: JPEG kvaliteet ümbersalvestamisel. OCR on teksti teravuse suhtes tundlik,
#: seega kõrge. Mõõdetud 21.07.2026 (10 lehe valim, aeg = dekodeeri+skaleeri
#: eelarvele, ehk see, mida protsessor treeningu ajal niikuinii teeb):
#:
#:   originaal 15 MP           100%   310 ms   1.0x
#:   q=92 subsampling=0         83%    38 ms   8.2x
#:   q=92 subsampling=2         75%    31 ms   9.9x   <- valitud
#:   q=88 subsampling=2         64%    29 ms  10.7x
#:
#: Alla 92 langeb maht veel, aga kiirus enam praktiliselt mitte – seega
#: pole mõtet kvaliteedis järele anda.
JPEG_QUALITY = 92

#: 4:2:0 kromasampling. Tekst on luminantsis, seega servad ei kannata;
#: värviline tint marginaalides pehmeneb marginaalselt.
JPEG_SUBSAMPLING = 2


def fit_to_budget(im: Image.Image, budget: int = MAX_PIXELS) -> Image.Image:
    """Skaleerib pildi eelarve piiresse. AINUS koht, kus geomeetria määratakse.

    KRIITILINE: seda peab kasutama nii treeningandmete ettevalmistamisel kui
    INFERENTSIL. Kui treening näeb LANCZOS-skaleeritud pilte ja inferents
    laseb protsessoril sama pildi BICUBIC-uga alla tõmmata, tekib
    treening/inferents-nihe. Mõõdetud 21.07.2026: LANCZOS vs BICUBIC on
    42-47 dB PSNR – sama suurusjärk kui JPEG q92 ümberkodeerimine (40-44 dB),
    ehk filtri valik EI ole tühine detail.

    Väiksemaid pilte ei suurendata: eelarve on lagi, mitte sihtmärk.
    """
    w, h = im.size
    if w * h <= budget:
        return im
    scale = (budget / (w * h)) ** 0.5
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def needs_resize(path: Path, budget: int = MAX_PIXELS) -> bool:
    """Kas pilt on eelarvest suurem?"""
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w * h > budget
    except Exception:
        return False


def prepare_image(src: Path, dst: Path, budget: int = MAX_PIXELS) -> str:
    """Kopeerib pildi väljundkausta, skaleerides eelarve piiresse.

    Treeningu ajal dekodeeritakse pilt kettalt iga epohhi kohta uuesti, ja
    protsessor skaleerib ta niikuinii eelarvele. Kui teha see juba siin, ei
    pea CPU sama tööd korduvalt tegema – meie skaneeringud on mediaanis ~3x
    eelarvest suuremad.

    Väiksemaid pilte EI suurendata: eelarve on lagi, mitte sihtmärk.

    Tagastab 'resized', 'copied' või 'kept' (dst oli juba olemas ja korras).
    """
    if dst.exists() and not needs_resize(dst, budget):
        return "kept"

    with Image.open(src) as im:
        w, h = im.size
        if w * h <= budget:
            shutil.copy2(src, dst)
            return "copied"
        fit_to_budget(im.convert("RGB"), budget).save(
            dst, "JPEG", quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING, optimize=True,
        )
    return "resized"
