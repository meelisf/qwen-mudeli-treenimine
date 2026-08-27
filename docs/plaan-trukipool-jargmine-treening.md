# Plaan: trükipoole järgmine treening

Koostatud 27.08.2026, llama.cpp juurdluse järelmina
(`docs/llamacpp-juurdlus-20260827.md`). Ei puuduta reedest Kurrendi-jooksu —
see on esimene järjekorras ja unslothi/transformersit ei tohi enne liigutada.

---

## 1. Pildieelarve — kaal LANGES 27.08.2026 õhtul

> **Parandus.** Lõikekatse näitas, et sama marginaaliveerg **täpselt samas
> pikslitiheduses** loeb eraldi pildina korrektselt (24 marginaali), terve lehe
> osana 0. Ehk teravus EI OLE piirang ja **eelarve tõstmine ei ole marginaalide
> lahendus** — ainult varu küsimus. Allolev analüüs jääb alles varu
> hindamiseks, aga see ei ole enam kõige olulisem punkt; selleks on 1b.

### Mida me nüüd teame

Marginaalirida on treeningresolutsioonis (2560×1952) **~43 px kõrge, x-kõrgus
~15 px**. Üks visuaaltoken katab 32×32 px, ehk **marginaalirida on 1,36 tokenit
kõrge**. Varu ei ole.

Ja me mõõtsime, kui vähe varu: kui llama.cpp kärpis pildi 4880 → ~3870 tokenile
(**9 % lineaarselt**), kadusid marginaalid **203 → 0**. Mitte ei halvenenud —
kadusid. Põhitekst ei kannatanud üldse. Ehk marginaalid istuvad täpselt
resolutsiooni põhjal, põhitekst mitte.

Eelarve on **andmestikku sisse küpsetatud**, mitte konfis:
`build_vutt_dataset.py` skaleerib pildid juba kettale (`prepare_image` →
`fit_to_budget`, LANCZOS). Mudel pole kunagi näinud lehte suuremana. Eelarve
muutmine tähendab andmestiku uuesti ehitamist.

### Kõva takistus: max_seq_length

`train_markup.py:212` – `UnslothVisionDataCollator(..., max_seq_length=8192)`.
Praegune järjend = transkriptsioon + ~5000 visuaaltokenit + 813 juhist.
Mõõdetud kõigi 1113 treeninglehe peal:

| | tokeneid |
|---|---|
| mediaan | 6893 |
| 90. protsentiil | 7442 |
| **max** | **8087** |
| max_seq_length | **8192** |

**Praegu ei kärbita ühtegi lehte — aga varu on 105 tokenit.** See on omaette
risk: veidi pikem transkriptsioon või veidi suurem pilt hakkab lehe lõppu
vaikselt ära lõikama.

Ja eelarve tõstmine murrab selle kohe:

| eelarve | visuaaltokeneid | lehti üle 8192 | vajalik max_seq_length |
|---|---|---|---|
| 5,12 Mpx (praegu) | ~5000 | 0 | 8192 |
| 6,0 Mpx | ~5860 | **236 (21 %)** | ≥ 8947 |
| 8,0 Mpx | ~7810 | **1113 (100 %)** | ≥ 10897 |
| 11,4 Mpx (2 tok/rida) | ~11130 | **1113 (100 %)** | ≥ 14217 |

`max_seq_length` 8192 valiti algselt just OOM-i tõttu (32768 ei mahtunud).
Ehk **pildieelarve tõstmine ei ole ajakulu küsimus, vaid mälu küsimus** ja
võib RTX 5090 peal üldse mitte mahtuda.

### Enne kui midagi otsustada: kaks katset, kokku ~20 min GPU-d

Mõlemad kasutavad **praegust mudelit**, treenida pole vaja.

**Katse A — kas mudel oskab neid tähti üldse lugeda?**
Lõika marginaaliveerg eraldi pildiks *originaalresolutsioonis* ja lase mudelil
see üksi transkribeerida.

- *Loeb õigesti* → mudel **oskab**, probleem on tähelepanu/eelarve jaotuses üle
  lehe, mitte glüüfide loetavuses. Siis on lahendus **lehe tükeldamine**, mitte
  globaalse eelarve tõstmine — ja tükeldamine on kordades odavam, sest iga tükk
  jääb lühikeseks ja `max_seq_length` ei muutu.
- *Ei loe* → glüüfid on päriselt piiri all, ja siis on vaja resolutsiooni,
  ehk kallist teed.

**Katse B — kas mudel oskab kasutada rohkem piksleid, kui talle anda?**
Sama 8 lehte inferentsil 8 Mpx juures (`--image-max-tokens 7810`). See on
treeningjaotusest väljas, aga näitab suunda.

- *Paraneb* → mudel üldistub ülespoole, eelarve tõstmine tõenäoliselt tasub
- *Halveneb* → OOD, kasu tuleb ainult uue treeninguga

**Ilma nende kaheta on iga eelarveotsus arvamus.** Kumbki ei maksa midagi peale
paarikümne minuti.

### Otsuspuu pärast katseid

| A | B | järeldus |
|---|---|---|
| loeb | ükskõik | **Tükelda leht** — odav, `max_seq_length` ei muutu, andmestikku ei pea ümber ehitama |
| ei loe | paraneb | Tõsta eelarvet 6 Mpx-ni (`max_seq_length` 8947, 21 % lehti puudutab) — väikseim samm, mis midagi annab |
| ei loe | halveneb | Resolutsioon ei ole ainus probleem; vaata treeningandmete küljendusjaotust (punkt 2) |

---

## 1b. LoRA maht ja kaheastmeline ahel — uus punkt (27.08.2026 õhtul)

### Asümmeetria on päris

| mudel | LoRA | treenitavaid | samme | loss lõpus | lähtepunkt |
|---|---|---|---|---|---|
| **trükk** 20260722 | r=16, α=16 | **51,0 M** | **268** | 0,036 | **eeltreenitud** (1500 lk transkriptsioon) |
| **Kurrent** 20260602 | r=64, α=64 | **203,9 M** | 3178 | 0,108 | puhas baas `unsloth/Qwen3.5-9B` |

Trükimudel teeb **raskemat** ülesannet — transkriptsioon *pluss* VUTT XML
märgendus — **neljandiku parameetritega** ja 12× väiksema sammuarvuga.

### Kaks asja, mis seda võrdlust nüansseerivad

**(1) Lossi ei saa otse võrrelda.** Trükk alustab soojalt: 1. etapi
transkriptsioonioskus on juba adapteris. Kurrent alustab külmalt. Trüki
loss 0,036 ei tõesta, et maht on piisav — ta tõestab, et mudel oskas juba
enne alustamist neid dokumente lugeda. (Selle vea tegin esimeses analüüsis.)

**(2) See r=16 adapter kannab KAHTE asja korraga.** `train_markup.py` laadib
`models/qwen3.5-ocr-lora-backup-20260527` ja treenib **sedasama adapterit
edasi** (`get_peft_model()` EI tohi järgneda). Ehk 51 M parameetrit hoiavad
nii 1. etapi transkriptsiooni kui märgendust; Kurrendi 204 M hoiavad ühte
asja. See on tugevam mahuargument kui pelk r=16 vs r=64.

### Tehniline takistus ja selle lahendus

**r-i ei saa olemasoleval adapteril tõsta.** r=16 adapterit ei saa r=64-na
edasi treenida. Kaks teed:

- **(a)** Treeni puhtast baasist r=64-ga korraga transkriptsioon + märgendus.
  Vajab mõlemat andmestikku ja on pikk jooks.
- **(b)** **Liida 1. etapi adapter baasi ja alusta värske r=64 adapteriga
  märgenduse peal.** See on nüüd võimalik, sest `scripts/merge_lora.py`
  sai täna valmis (GGUF-i jaoks, aga sobib täpselt siia):

  ```bash
  venv/bin/python scripts/merge_lora.py models/qwen3.5-ocr-lora-backup-20260527
  venv/bin/python scripts/train_markup.py \
      --base=models/merged/qwen3.5-ocr-lora-backup-20260527-bf16 --lora-rank=64
  ```
  Siis kannab adapter ainult märgendust, transkriptsioon on kaaludes sees.

  **KOODIMUUDATUS ON VAJALIK, enne kui see käsk töötab.** `train_markup.py`-l
  ei ole `--lora-rank` lippu ega `get_peft_model()` kutset üldse — ta OSKAB
  ainult olemasolevat adapterit edasi treenida. `train_kurrent.py`-s on
  täpselt õige muster juba olemas ja sealt saab selle üle tuua:

  | mida | `train_kurrent.py` |
  |---|---|
  | `LORA_RANK` konstant + `--lora-rank=` lipp | read 57, 65–66, 72 |
  | tingimuslik `get_peft_model()` HF-baasi puhul | read 74, 129–139 |

  Töö on väike (~20 rida), aga seda ei tohi teha „muuseas" — see muudab
  `train_markup.py` käitumist ka tavajooksul, seega tuleb tingimus kirjutada
  nii, et olemasolev ahel (checkpointist edasi) käitub täpselt nagu praegu.

### Aga ilma holdout'ita ei saa seda mõõta

Trüki-holdout'i ei ole, ehk r=16 vs r=64 võrdlust ei saaks tõestada — teeksime
kalli jooksu ja vaataksime tulemust silmaga, nagu täna. Ja andmenappus on
sõltumatu probleem: **Menii on treeningus 0 lehte**, ≥30 marginaaliga lehti on
41 ja ≥39 marginaaliga **3**, samas kui katkised lehed on 24–39 vahemikus.
Mudel on seda otsust näinud kolm korda.

**Järjekord:** (1) 10-leheline holdout, (2) 20–30 Menii-tüüpi lehte
märgendatud, (3) alles siis r=16 vs r=64 kontrollitud võrdlusena tee (b) kaudu.
Mitte sellepärast, et maht oleks vale hüpotees, vaid sellepärast, et ilma
esimese kaheta ei saa tulemust tõestada.

---

## 2. Marginaalirohke materjal — tööjärg, mitte ajakava

**Seis:** marginaalide kogus treeningus ei ole kõhn — 527 lehte 1113-st (47 %),
8505 `<m>` tagi, 65 teosest. Kitsas on **tihedus ja žanr**:

| `<m>` tagi lehel | treeninglehti |
|---|---|
| ≥10 | 383 |
| ≥25 | 107 |
| ≥30 | **41** |
| ≥39 | **3** |

Katkised Menii lehed on 24–39 tagiga, ehk jaotuse 90.–100. protsentiilis.
Ja **Frid. Menii ei ole treeningandmetes üldse** (0 lehte). Treening on
valdavalt Academia disputatsioonid ja oratsioonid, max 12 lehte teose kohta;
Menii on ajalooraamat tiheda ääreveeruga — **tundmatu küljendus**, mitte lihtsalt
tundmatu teos.

**Piirang:** doktorandid töötavad oma materjalidega, seda järjekorda ei muuda.
Menii teeb kasutaja ise, kui tunde jagub. Ehk see punkt **ei ole ajastatav** —
aga kui tunnid tulevad, on olemas prioriteedireegel, mida enne polnud:

> **Anna inimesele esimesena need lehed, kus kaks mudeliahelat lahku lähevad.**

Nimekiri on olemas: `docs/marginaalid-silmaga-vaadata.md` — 23 lehte,
vaidlusalused read kõrvuti. Neist 143 re-OCR-lehest on **59 just Menii omad**.
Seal on mudel ebakindel ja sisu rikas; iga kinnitatud leht parandab täpselt seda,
mis praegu katki on. See on aktiivõpe, mis ei maksa midagi peale inimtunni.

---

## 3. Holdout — 10 lehte, ja üks tasuta lisakiht

**Piirang:** trükimaterjal on kallis, rohkem kui **10 lehte** ei taha treeningust
välja jätta. Nõus — ja seda saab targemini teha, kui esmapilgul paistab.

### Kaks kihti, erineva hinnaga

**Kiht 1 — 10 lehte päris ground truth'iga (maksab 10 treeninglehte).**
Mõõdab **absoluutset** kvaliteeti: CER, marginaalide täpsus. Valikureegel:
võta **tüüpilisi, mitte haruldasi** lehti. Haruldase küljenduse väljajätmine
maksab treeningus rohkem kui tüüpilise. Jaotus umbes: 4 marginaalirohket
(≥20 `<m>`), 2 marginaalideta, 2 kaldkirja/`<cs>`-rohket, 1 tabel/register,
1 hõre või tühjapoolne leht.

**Kiht 2 — ~143 „Toores" lehte, ilma GT-ta (maksab MITTE MIDAGI).**
Need ei ole niikuinii treeningus, sest inimene pole neid kinnitanud. Nad ei
mõõda absoluutset kvaliteeti, aga mõõdavad **regressiooni**: kas uus mudel
läheb vana suhtes kuskil katki. Täpselt see, mille jaoks `reocr_vutt.py`
tänase juurdluse käigus valmis sai.

Ehk: **10 lehte ütlevad, kui hea mudel on; 143 lehte ütlevad, kas ta läks
katki.** Teine kiht kasvab iseenesest iga kord, kui teenus uusi lehti töötleb.

### Millal

Kiht 2 on juba olemas ja töötab. Kiht 1 tasub teha **enne järgmist
trükitreeningut**, mitte enne seda — praegu pole midagi mõõta, ja 10 lehte
treeningust välja võtta ilma põhjuseta oleks puhas kaotus.

---

## Kokkuvõttes, järjekorras

| # | samm | maksumus | blokeerib |
|---|---|---|---|
| 0 | Reedene Kurrendi-jooks | — | kõike muud |
| 1 | Katsed A ja B | ~20 min GPU | punkti 1 otsust |
| 2 | Teenuse BICUBIC-nihe ära parandada | ~1 h, ei vaja treeningut | mitte midagi (tasuta ~7 % marginaale) |
| 3 | Otsus pildieelarve / tükeldamise kohta | sõltub A ja B tulemusest | järgmist treeningut |
| 4 | 10-leheline GT-holdout | 10 treeninglehte | ainult mõõtmist |
| 5 | Menii ja vaidluslehtede märgendamine | inimtunnid, ajastamata | — |

Punkt 2 on ainus, mis on **kohe ja tasuta**: teenus laseb protsessoril BICUBIC-uga
skaleerida, treeningandmed valmistati LANCZOS-iga, vahe on mõõdetult ~7 %
marginaale. `kataloogi-jalgimine-ja-ocr.py` peaks kutsuma
`imaging.fit_to_grid()` enne protsessorit.
