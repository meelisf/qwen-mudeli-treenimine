#!/usr/bin/env python3
"""
Kordusloopi tuvastus väljundtekstist – jagatud loogika.

Mõõdetud 27.08.2026 (vt mälu „llamacpp-gguf-konversioon"): loop EI OLE ühegi
konkreetse konfiguratsiooni omadus. Sama mudel loopis eri lehtedel sõltuvalt
sellest, kas kasutati unslothi või llama.cpp-d, Q8_0 või BF16 kvantimist,
LANCZOS või BICUBIC skaleerimist. Ükski kombinatsioon ei olnud loobivaba ja
samplerid (DRY, repeat_penalty) tegid asja hullemaks. Ehk **loopi ei saa ära
konfigureerida, ainult tuvastada** – ja tuvastada saab sama hästi pärast
genereerimist kui selle ajal.

`find_tail_loop` on sama algoritm, mis teenuse `LoopStopper`-is
(kataloogi-jalgimine-ja-ocr.py, VUTT #227). Erinevus: seal jookseb ta
genereerimise ajal tokenite sabal, siin valmis teksti peal.

NB! Teenuses on praegu oma koopia. Kui see moodul jääb kasutusse, tuleks
teenus siia peale lülitada, muidu jooksevad kaks koopiat lahku.
"""

#: Perioodi ülempiir SÕNADES. Teenuses on praegu 20 (commit f0058b5, tõsteti
#: 5 -> 20). **20 on liiga väike:** mõõdetud 27.08.2026, päris loop
#: `12740_bergskollegium_adv` on perioodiga **26 sõna** (48 kordust) ja jääb
#: 20 juures märkamata. Kalibreerimine 438 Kurrendi väljundi peal:
#:
#:   max_period   raskeid loope tabab   valehäireid (419 tervet lehte)
#:           20             4 / 8                    0
#:           30             7 / 8                    0
#:           80             7 / 8                    0
#:
#: 30 peale tõstmine ei maksa midagi ja tabab pea kaks korda rohkem. Üle 30
#: ei anna juurde. Ainus märkamata juhtum oli DRY-samplerist tulnud
#: LÄHEDANE kordus ("Anno 1707" -> "Anno 2230"), mida täpne tsüklikontroll
#: põhimõtteliselt ei näe – veel üks põhjus DRY-d mitte kasutada.
LOOP_MAX_PERIOD = 30
LOOP_MIN_REPS = 3


def find_tail_loop(sonad, max_period=LOOP_MAX_PERIOD, min_reps=LOOP_MIN_REPS):
    """Kas sõnajärjendi LÕPP on korduv tsükkel? Tagastab (periood, kordused) või None.

    Vaatab ainult saba: nii ei anna varem lõppenud kordus (loetelu 'I. II. III.')
    valehäiret.
    """
    n = len(sonad)
    for period in range(1, max_period + 1):
        if n < period * min_reps:
            continue
        muster = sonad[n - period:]
        reps = 1
        i = n - 2 * period
        while i >= 0 and sonad[i:i + period] == muster:
            reps += 1
            i -= period
        if reps >= min_reps:
            return period, reps
    return None


#: Märgitasandi korduse parameetrid. Sõnapõhine `find_tail_loop` EI NÄE
#: kordust, milles pole tühikuid – terve degenereerunud saba on tema jaoks
#: üks pikk "sõna". Mõõdetud 27.08.2026 trüki-re-OCR-il: 3 lehte 83-st
#: degenereerusid just nii ('ææææ…' ja heebrea tähekordus 1638-39 Grammatica
#: Ebraea lehtedel) ja jäid sõnadetektorile märkamata.
CHAR_TAIL = 400           # mitu viimast märki vaadata
CHAR_MAX_PERIOD = 40
CHAR_MIN_REPS = 4


def find_char_loop(text, tail=CHAR_TAIL, max_period=CHAR_MAX_PERIOD,
                   min_reps=CHAR_MIN_REPS):
    """Kas teksti lõpp on korduv MÄRGIMUSTER? Tagastab (periood, kordused) või None."""
    s = text.rstrip()[-tail:]
    n = len(s)
    for period in range(1, max_period + 1):
        if n < period * min_reps:
            continue
        muster = s[n - period:]
        reps = 1
        i = n - 2 * period
        while i >= 0 and s[i:i + period] == muster:
            reps += 1
            i -= period
        if reps >= min_reps:
            return period, reps
    return None


def is_looped(text: str):
    """Kas valmis väljund lõpeb kordustsükliga? Tagastab (periood, kordused) või None.

    **Viimane sõna visatakse ära.** Loopinud väljund lõpeb tokenilaes keset
    sõna ja see poolik sõna lõhub tsükli – ilma selleta ei tuvasta detektor
    mitte ühtegi päris loopi. Teenuse `LoopStopper` teeb sedasama
    (`saba.split()[:-1]`), sest kontroll langeb keset tokenit.

    Mõõdetud 27.08.2026, 438 Kurrendi väljundit:
      - ratio > 2,0 (päris loop):     8 juhtu, tabab 7 (88 %)
      - ratio 1,4-2,0:               11 juhtu, tabab 0 – ja need EI OLE loobid,
        vaid lehed, mille arhiivi-GT on lühem kui leht ise (nt 13220 lõpeb
        korraliku tekstiga). Ehk „ratio > 1,4 = loop" ÜLEHINDAB loopide arvu.
      - ratio <= 1,4 (terved):      419 juhtu, tabab 0 (valehäireid ei ole)
    """
    return find_tail_loop(text.split()[:-1]) or find_char_loop(text)
