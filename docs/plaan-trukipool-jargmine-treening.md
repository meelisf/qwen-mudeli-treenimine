# Plaan: trükipoole järgmine treening

Koostatud 27.08.2026, llama.cpp juurdluse järelmina
(`docs/llamacpp-juurdlus-20260827.md`). Ei puuduta reedest Kurrendi-jooksu —
see on esimene järjekorras ja unslothi/transformersit ei tohi enne liigutada.

---

## 1. Pildieelarve — kõige olulisem, ja kõige rohkem takistusi

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
