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


def fit_to_budget(im: Image.Image, budget: int = MAX_PIXELS,
                  resample: int = Image.LANCZOS) -> Image.Image:
    """Skaleerib pildi eelarve piiresse. AINUS koht, kus geomeetria määratakse.

    KRIITILINE: seda peab kasutama nii treeningandmete ettevalmistamisel kui
    INFERENTSIL. Kui treening näeb LANCZOS-skaleeritud pilte ja inferents
    laseb protsessoril sama pildi BICUBIC-uga alla tõmmata, tekib
    treening/inferents-nihe. Mõõdetud 21.07.2026: LANCZOS vs BICUBIC on
    42-47 dB PSNR – sama suurusjärk kui JPEG q92 ümberkodeerimine (40-44 dB),
    ehk filtri valik EI ole tühine detail.

    Väiksemaid pilte ei suurendata: eelarve on lagi, mitte sihtmärk.

    `resample` on olemas AINULT selleks, et seda nihet saaks mõõta
    (`eval_kurrent.py --resample bicubic` jäljendab image processori enda
    skaleerimist). Andmestiku ettevalmistuses jäta see puutumata.
    """
    w, h = im.size
    if w * h <= budget:
        return im
    scale = (budget / (w * h)) ** 0.5
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)


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


#: Patch-võre samm: patch_size 16 x spatial_merge_size 2. Qwen3.5 protsessor
#: ümardab pildi mõlema külje selle kordseks.
GRID_FACTOR = 32


#: Alumine pikslipiir – protsessori `shortest_edge`. Väiksem pilt suurendatakse.
MIN_PIXELS = 65536


def fit_to_grid(im: Image.Image, budget: int = MAX_PIXELS,
                factor: int = GRID_FACTOR,
                resample: int = Image.LANCZOS,
                floor: int = MIN_PIXELS) -> Image.Image:
    """Skaleerib pildi TÄPSELT sellele võrele, mida Qwen3.5 protsessor valiks.

    Sama aritmeetika mis HF `smart_resize` (Qwen2VL image processor): mõlemad
    küljed ümardatakse `factor`-i kordseks ja kui tulemus ületab eelarve,
    tõmmatakse alla ja ümardatakse allapoole.

    MILLEKS: llama.cpp teeb sedasama ümardamist ise, aga **ilma
    antialiasinguta** (`image_resize_pad = PAD_CEIL`, ülesvoolu PR #17577
    parandas selle ainult LFM2-VL-ile). Väike allaskaleerimine ilma
    antialiasinguta hävitab õhukesed kaldkirjatähed – mõõdetud 27.08.2026:
    marginaaliveerg kadus tervetel lehtedel. Kui klient annab pildi juba
    õigel võrel, ei ole llama.cpp-l midagi skaleerida.

    NB! Saada tulemus PNG-na, mitte JPEG-ina. Lähtefail on juba JPEG ja teine
    JPEG-põlvkond sööb needsamad õhukesed tähed ära (mõõdetud: `<m>` tage 76
    JPEG-iga vs 150 PNG-ga samal 8 lehel).
    """
    import math

    w, h = im.size
    w_bar = max(factor, round(w / factor) * factor)
    h_bar = max(factor, round(h / factor) * factor)
    if w_bar * h_bar > budget:
        beta = (w * h / budget) ** 0.5
        w_bar = max(factor, int(w / beta / factor) * factor)
        h_bar = max(factor, int(h / beta / factor) * factor)
    elif w_bar * h_bar < floor:
        # Alumine piir: protsessor suurendab liiga väikese pildi üles. Meie
        # skaneeringud siia ei satu, aga ilma selleta lahkneb funktsioon
        # protsessorist (mõõdetud: 4 juhtu 1120-st, kõik alla 130 px küljega).
        beta = (floor / (w * h)) ** 0.5
        w_bar = max(factor, math.ceil(w * beta / factor) * factor)
        h_bar = max(factor, math.ceil(h * beta / factor) * factor)
    if (w_bar, h_bar) == (w, h):
        return im
    return im.resize((w_bar, h_bar), resample)
