# Kadunud marginaalid

**Miks Qwen3.5-9B OCR-mudel kaotas llama.cpp peal terve marginaaliveeru.**
Juurdlus 27.08.2026. llama.cpp b10641 (539f24529), RTX 5090 @ 450 W.

> **Versioon 2.** Esimene versioon sisaldas kolme viga, mille leidis
> vastastikune ülevaatus (vt „Parandused v1 suhtes" lõpus). Kõik numbrid siin
> on üle kontrollitud.

Mudel viidi llama.cpp peale, et inferents kiiremaks saada. Kurrendi
käsikirjadel läks kõik hästi. Trükilehtedel kadus kaheksal leheküljel *terve
marginaaliveerg* — ja kaalud olid bitipealt õiged. Kaks põhjust, mõlemad
seadistuses; üks neist meie enda kliendis.

---

## Mida taheti

Praegune tootmisahel on unsloth + transformers. Küsimus: kas mootori vahetus
llama.cpp peale annab kiirust ilma täpsust kaotamata. Vahesammud on tavalised:

```
LoRA adapter → merge_and_unload → merged BF16 → convert_hf_to_gguf
             → Q8_0 + mmproj → llama-server
```

---

## Lõpptulemus

**Kurrent (73-leheline holdout):**

| variant | s/lk | lk/h | CER kõik | CER 69 puhtal | mediaan | loope |
|---|---|---|---|---|---|---|
| unsloth bf16, batch 4 (etalon) | 19,6 | 183 | 13,9 % | **8,7 %** | 6,4 % | 2 |
| llama.cpp Q8_0, vaikeseaded, JPEG | 3,5 | 1031 | 15,6 % | 9,2 % | 6,4 % | 3 |
| llama.cpp Q8_0, 5120, JPEG | 4,5 | 807 | 13,9 % | 8,7 % | 6,0 % | 3 |
| **llama.cpp Q8_0, 5000 + PNG + 32×** | **4,7** | **767** | **13,9 %** | **8,8 %** | **6,1 %** | 3 |

Lehe kaupa vs unsloth: **19 paremat, 15 halvemat, 35 sama.** Pariteet.
GPU-mälu 12,7 GB vs 25,2 GB. **4,2× kiirem.**

> Varasem „5,6× kiirem" oli mõõdetud kärbitud pildiga (vt Põhjus 1).
> Täisresolutsioonis on võit 4,2×.

**Trükk (143 VUTT-i „Toores" lehte, 134 puhast):**

| | lahknevus (mediaan) | `<m>` | `<i>` | `<cs>` | marginaalid kadusid |
|---|---|---|---|---|---|
| JPEG + vaikepiir | 4,9 % | 634 → 527 | 1120 → 1094 | 131 → 66 | 7 lehel |
| **PNG + 5000 + 32×** | **3,0 %** | 634 → **684** | 1120 → **1177** | 131 → 60 | **3 lehel** |

Marginaale tuleb nüüd **rohkem** kui praegusest teenusest. `<cs>` jääb aga
poole peale — see on eraldi, seni seletamata erinevus.

---

## Kurrent: mootorivahetus töötab

Unslothi jooks on korratav — kordusmõõtmine andis 73/73 bait-baidilt sama
väljundi mis päev varem, seega mootorite vahe on päris vahe, mitte jooksumüra.

### Samplerid loobide vastu: mõlemad tagasi lükatud

| variant | CER kõik | CER puhtad | loope |
|---|---|---|---|
| ilma samplerita | 15,6 % | 9,2 % | 3 |
| DRY 0,8 | 26,2 % | 9,0 % | **4** |
| repeat_penalty 1,1 | 11,4 % | **9,6 %** | 2 |

**DRY tekitas uue loobi** lehel, mis oli varem terve, ega parandanud seda, mille
pärast teda proovitigi. Halvem veel: DRY muudab täpse korduse *ligikaudseks*
(„Anno 1707" → „Anno 2230"), mida tsüklidetektor põhimõtteliselt ei näe.

**repeat_penalty 1,1 koond-CER 11,4 % on keskmistamise pettus.** Kogu võit tuleb
ühest loopinud lehest (380 % → 20,5 %). Puhastel lehtedel teeb ta halvemaks —
19 lehte halvemaks, 10 paremaks — ja kaks lagunevad täiesti (3,6 % → 29,4 %;
14,9 % → 32,7 %), mõlemal `ratio` ~1,20. Kordusekaristus surub alla ka
lõpetamise.

**Otsus: samplerit ei muuda.** Loop tuvastatakse järelkontrolliga ja märgitakse
`.err`-iga. Kulu on piiratud: llama.cpp-s maksab lakkejooksnud leht 87 s ja
mõjutab ainult ennast; unslothi batch 4 puhul maksis sama loop 330 s ja hoidis
kolme süütut lehte kinni.

### Loobidetektor oli vale häälestusega

Teenuse `LoopStopper` otsib kuni 20 sõna pikkust tsüklit. Päris loobi periood
mõõdeti üle: **26 sõna**, 48 kordust.

Kalibreerimiskorpus: **438 väljundit = 419 normaalset + 11 GT-mismatch
mitte-loopi + 8 päris loopi.** Need 11 on lehed, mille `ratio` on 1,4–2,0 —
mitte loobid, vaid lehed, mille arhiivi-GT on lühem kui leht ise. Senine
„`ratio` > 1,4 = loop" ülehindas loopide arvu enam kui kaks korda.

| max_period | tabab 8 päris loobist | valehäireid **kõigi 430 mitte-loobi peal** |
|---|---|---|
| 20 (praegune teenuses) | **4** | 0 |
| 30 | **7** | 0 |
| 80 | 7 | 0 |

Kaks parandust veel:

- **Viimane sõna tuleb ära visata** — loopinud väljund lõpeb tokenilaes keset
  sõna ja see poolik sõna lõhub tsükli; ilma selleta ei tuvasta järelkontroll
  ühtegi päris loopi.
- **Märgitasandi kordus** (`ææææ…`, heebrea tähekordus) jääb sõnadetektorile
  nähtamatuks, sest tühikuteta saba on tema jaoks üks pikk sõna — 3 lehte 83-st
  degenereerusid just nii.

**Valehäirete kontroll päris andmetel.** 1189 inimese kinnitatud (`Valmis` /
`Parandatud`) trükiteksti peal annab detektor **1 valehäire**: pühendusvormel
`D. D. D.` (*Dat Dicat Dedicat*) on periood 1, kolm kordust. Sünteetilised
tüpograafilised mustrid näitavad sama riski laiemalt:

| muster | detektor |
|---|---|
| `-----` korduvad eraldusjooned | **märgib** (valehäire) |
| sisukorra punktiir `1. . . . 5` | **märgib** (valehäire) |
| `I. II. III. IV. V.` | ei märgi |

Lühikeste perioodide (1–2 sõna) puhul tuleks nõuda rohkem kordusi — päris
loobid olid (1, 1365), (1, 809), (1, 9) ja (26, 48), ehk lävi ≥8 kordust
säilitaks kõik päris juhtumid ja välistaks `D. D. D.`

---

## Trükipool: siin läks midagi katki

Trükimudelil ei ole ground truth'i, aga VUTT-i backup-snapshotis on 143 lehte,
mille staatus on `Toores`, mille praegune tootmismudel on juba transkribeerinud
ja mida ükski inimene pole puutunud. Vana väljund on kettal olemas — tasuta A/B.

35 lehel, kus vanas väljundis oli vähemalt viis `<m>` märgendit, kadusid need
**kaheksal täielikult**. Kõik kaheksa ühest teosest: *Frid. Menii Syntagma de
origine Livonorum* (1635), kus marginaalid on kitsas ääreveerus väikeses
kaldkirjas.

**Ja see ei olnud märgendusprobleem, vaid nägemisprobleem.** Lehel
`1635_1_0028` oli vanas väljundis 39 marginaalirida. Uues väljundis ei leidunud
neist ühtegi *isegi märgendamata tekstina*. Mudel transkribeeris ainult
põhiveeru ja lõpetas korralikult, signatuuriga „B".

### Väljundi tokenilagi on eraldi asi

Selles juurdluses on **kaks erinevat 4096-piiri** ja neid on lihtne segi ajada:

| | mis see on | mõju |
|---|---|---|
| **sisendi 4096** | `clip.cpp` visuaaltokenite lagi | pilt kärbitakse vaikselt → Põhjus 1 |
| **väljundi 4096** | meie `max_tokens` genereerimisel | leht võib jääda pooleli |

Väljundi lakke jooksis puhtas jooksus **6 lehte 143-st (4 %)**, JPEG-jooksus
8 lehte (6 %). Klassifitseerituna (JPEG-jooks): **5 loopi**, **3 mitte-loopi** —
neist üks päriselt liiga pikk leht (`1654_20_41_43_Image_003`, GT 7440 märki,
väljund 9792, kärbub mõlemas ahelas), üks märgitasandi degeneratsioon
(`1633_2_0002`, `æææ…`) ja üks piiripealne (`1638_39_0133`).

---

## Välistamisahel

Küsimus, mis suunas ülejäänud töö: kui tegemist on tavalise peenhäälestusega,
miks marginaalide märgendamine GGUF-i kaasa ei tule? Vastus: tulebki kaasa.
Viga oli mujal.

| # | seis | kahtlus | tõend |
|---|---|---|---|
| 1 | välistatud | LoRA ei puutunud visuaaltorni | Puutus: 712 adaptertensorist 216 on `model.visual.blocks.*` |
| 2 | välistatud | Liitmine ei rakendunud | Rakendus: merged vs baas, visuaal max\|Δ\| 0,0023, keel 0,0017 |
| 3 | välistatud | Konversioon kaotas visuaalkaalud | mmproj GGUF vs merged: max\|Δ\| **0,000000** — bitipealt identsed |
| 4 | välistatud | F16 mmproj kaotas täpsust | max\|Δ\| originaalist 3·10⁻⁸; suurim kaal 0,582, kaugel f16 piirist |
| 5 | välistatud | Viga on liitmises, mitte mootoris | Merged mudel transformersis säilitab marginaalid (0038: 27 märgendit) |
| 6 | välistatud | Q8_0 kvantimine | **Lõpliku ahelaga**: BF16 GGUF 156 vs Q8_0 156, lehe pealt identne |
| 7 | välistatud | Resample-filter | LANCZOS 130 vs BICUBIC 115 — samad lehed kaovad mõlemaga |
| 8 | välistatud | Serva lõikamine 32-ümardamisel | Transformers ümardab samamoodi: 2588×1978 → 2560×1952 mõlemal |
| 9 | välistatud | Chat template erineb | `diff`-ga identne: checkpoint == ametlik baas == merged |
| 10 | välistatud | Teksti/pildi järjekord erineb | Treeningul `text` enne `image`; päringus sama |
| 11 | välistatud | mrope-sektsioonid valed | `config.json` [11,11,10] → konverter padib [11,11,10,0]; vastab |
| 12 | välistatud | `--no-nextn` võtab midagi ära | Eemaldab ainult MTP-kihid; pole autoregressiivse genereerimise osa |
| 13 | välistatud | Positsioonikodeeringu interpolatsioon | llama.cpp `ALIGN_CORNERS` vastab HF `fast_pos_embed_interpolate`-le |
| 14 | **PÕHJUS 1** | llama.cpp piirab pildi 4096 visuaaltokenini | `clip.cpp`: `set_limit_image_tokens(8, 4096)` — meie eelarve on 5000 |
| 15 | **PÕHJUS 2** | Klient kodeeris pildi JPEG-ina uuesti | Teine JPEG-põlvkond; transformersi teel ei toimu ümberkodeerimist |
| 16 | *lahtine* | 2 lehte 8-st kaotavad marginaalid ka lõpliku ahelaga | Ja `<cs>` jääb poole peale |

### Punkt 6 väärib eraldi märkust

Q8_0 välistati **kaks korda**. Esimene katse tehti enne lõpliku pildiahela
leidmist ja oli seetõttu nõrk tõend: kui lehed on noateral, võib kvantimise
mõju tulla nähtavale alles siis, kui pildiinfo on taastatud. Kordus lõpliku
ahelaga (PNG + 32× + 5000):

| leht | VUTT | unsloth | Q8_0 PNG | BF16 PNG |
|---|---|---|---|---|
| 0012 | 14 | 31 | 37 | 37 |
| 0017 | 28 | 21 | 0 | 0 |
| 0024 | 24 | 24 | 26 | 26 |
| 0028 | 39 | 39 | 39 | 39 |
| 0030 | 27 | 27 | 27 | 27 |
| 0036 | 7 | 10 | 0 | 0 |
| 0038 | 27 | 27 | 27 | 27 |
| **KOKKU** | **166** | **179** | **156** | **156** |

Lehe pealt identne. Kvantimine ei ole jääkpõhjus.

---

## Põhjus 1 — vaikne pildipiir

`tools/mtmd/clip.cpp`, `case PROJECTOR_TYPE_QWEN3VL`:

```cpp
hparams.set_limit_image_tokens(8, 4096);
```

Maksimaalselt 4096 visuaaltokenit pildi kohta, sõltumata sellest, mida mudel
treeningul nägi. Meie eelarve on 5 120 000 px ehk **5000 tokenit** (pikslid ÷
1024). llama.cpp skaleerib pildi seetõttu veel kord alla — juba niigi ~3×
vähendatud skaneeringu peal — ja teeb seda vaikides: logis pole selle kohta
rida.

Lipp on `--image-max-tokens N`. Mõju: prompt-tokenid 4842 → 5849, ja kaheksast
kadunud lehest tuli viis tagasi.

**Kuidas geomeetriat kontrollida.** Prompt-tokenite vahe tekst-ainult ja
pildiga päringu vahel annab pildi lisandumise mõju, mitte matemaatiliselt
patchidest tulenevate embeddingute arvu — Qweni mall lisab pildiga ka
wrapper-tokeneid (meil +2). Usaldusväärne on **kaks mõõtu koos**:
deterministlik valem `(H/32) × (W/32)` ja prompt-tokenite vahe. Meie juhul
4880 vs mõõdetud 4882.

Juhise pikkust silma järgi hinnata ei tohi: eksisin 145 tokeniga ja järeldasin
sellest ekslikult, et geomeetria klapib.

### See ei ole meie mudeli eripära — ega Unslothi lahendatud

`gguf-py/gguf/gguf_writer.py`-s on `add_vision_image_max_pixels` olemas, aga
**llama.cpp konversioonikood ei kutsu seda kordagi** (kontrollitud b10641).

Kontrollisin ka Unslothi ametlikku eksporti, laadides
`unsloth/Qwen3.5-9B-GGUF/mmproj-F16.gguf` päise (3 MB range-päring; GGUF-i
metaandmed on faili alguses):

```
metaandmete võtmeid: 32, neist clip.*: 15
clip.vision.image_size          LEIDUB   ← otsimeetod töötab
clip.vision.patch_size          LEIDUB
clip.vision.image_min_pixels    EI LEIDU
clip.vision.image_max_pixels    EI LEIDU
```

Sama 15-võtmeline `clip.*` komplekt mis meil — sama konverteritee. **Unsloth
Studio ei ole seda lahendanud.** Ehk meil ei ole valmis retsepti, aga on
konkreetne veakirjeldus ülesvoolu: iga Qwen3-VL / Qwen3.5 GGUF selle teega on
4096 tokeni peal, kui kasutaja lippu ei tea.

---

## Põhjus 2 — teine JPEG-põlvkond

HTTP-klient kodeeris eelarvest suurema pildi JPEG-ina uuesti (q92) enne base64
saatmist. Lähtefail on juba JPEG, ehk see on teine põlvkond. Transformersi teel
ei toimu ühtegi ümberkodeerimist — pilt liigub PIL-objektina otse protsessorile.
Õhukesed kaldkirjaga marginaalitähed on täpselt see, mida teine JPEG-põlvkond
ära sööb. **See oli meie enda viga, mitte llama.cpp oma.**

Kaheksa lehte, `<m>` märgendite arv (loopinud leht 0025 välja jäetud):

| variant | `<m>` kokku | lehti märgendiga |
|---|---|---|
| VUTT (praegune teenus) | 166 | 7 / 7 |
| unsloth | 179 | 7 / 7 |
| llama.cpp, vaikepiir 4096, JPEG | **0** | 0 / 7 |
| llama.cpp 5120, JPEG | 130 | 5 / 7 |
| llama.cpp 5000, JPEG | 52 | 3 / 7 |
| llama.cpp 5000, JPEG, 32× | 76 | 3 / 7 |
| **llama.cpp 5000, PNG, 32×** | **156** | **5 / 7** |

### Üks kontraintuitiivne mõõtmine

`--image-max-tokens 5000` viib geomeetria transformersiga kokku (4882 vs 4880),
aga andis JPEG-iga **halvema** tulemuse kui lõtvem 5120. Põhjus: 5120 juures
skaleerib llama.cpp pilti pisut *üles* (2588→2592), 5000 juures *alla*
(2588→2560), ja allaskaleerimine ilma antialiasinguta hävitab õhukesed tähed.
llama.cpp `image_resize_pad` on Qwen-harus `PAD_CEIL` ja antialias puudub;
ülesvoolu parandati see LFM2-VL jaoks (PR #17577), Qwen-perekonda ei laiendatud.

Sellest tuli lahendus: `imaging.fit_to_grid()` — klient skaleerib pildi ise
täpselt sellele patch-võrele, mida Qwen3.5 protsessor valiks, LANCZOS-iga, ja
saadab PNG-na. Siis pole llama.cpp-l midagi skaleerida ega uuesti pakkida.

**Property-test:** `fit_to_grid()` võrreldi HF protsessori valitud mõõtudega
**1240 juhusliku ja piiripealse mõõdu peal** (väga kitsad, väga laiad, täpselt
eelarve piiri ümbrus, 32-kordsed, pisipildid) — **0 lahknevust**. Esimene
versioon lahknes 4 juhul, kõik alla 130 px küljega piltidel, sest alumine
`min_pixels` piir oli implementeerimata.

---

## Kaks etaloni, mis omavahel ei klapi

VUTT-i salvestatud väljund annab neil kaheksal lehel **166** märgendit, minu
otsene unsloth-jooks **179**. See on 13-märgendine erinevus **enne llama.cpp-d**
ja see tuleb ära seletada, enne kui jääkvahet taga ajada. Seletus on olemas:

| ahel | `<m>` |
|---|---|
| VUTT (teenus: toorpilt → processor BICUBIC) | 203 * |
| minu unsloth + **toorpilt** (teenuse kordus) | 200 * |
| minu unsloth + **LANCZOS** eelskaleering | 216 * |

<small>* koos loopinud lehega 0025; ilma selleta 166 / — / 179.</small>

Teenus ja tema kordus klapivad (203 vs 200). Vahe tuleb **eelskaleerimisest**:
teenus laseb protsessoril BICUBIC-uga skaleerida, treeningandmed valmistati
ette LANCZOS-iga (`build_vutt_dataset.py` → `prepare_image`). Ehk **praegune
tootmisteenus on ise treeningust nihkes** ja jätab ~7 % marginaale leidmata.
Uus ahel parandab selle kõrvalproduktina.

---

## Kumb on parem? Kumbki pool ei ole tõde

`39 = 39` on hea struktuurne signaal, aga mudel võib toota sama palju `<m>`
ridu vale tekstiga. Seepärast mõõdeti **marginaalide sisu** eraldi: `<m>`
sisu ekstraheeriti mõlemast ja võrreldi editdistance'iga, 34 lehel, kus
VUTT-i väljundis oli ≥5 marginaali ja kumbki pool ei loopinud.

| | |
|---|---|
| mediaan lahknevus | **9,6 %** |
| alla 10 % | 17 lehte |
| alla 25 % | 21 lehte |
| üle 60 % | 8 lehte |

Mitmel lehel on marginaalid **bait-baidilt identsed** (0,0 %).

**Ja suure lahknevusega lehed ei ole vead.** Kolmel kõige kaugemal lehel
sisaldab uus väljund **100 % vanadest marginaaliridadest** (14/14, 9/9, 6/6)
ja lisaks neid, mida vana ahel ei näinud. Lahknevus tuleb täielikkusest, mitte
eksimusest.

Tervikpilt 41 lehel, kus marginaale üldse on:

| | lehti |
|---|---|
| uus leidis **juurde** ega kaotanud midagi | 11 |
| uus **kaotas** midagi | 12 |
| täpselt sama | 18 |
| **ridu juurde 137, kaotatud 88 → neto +49** | |

Ehk llama.cpp leiab 137 marginaalirida, mida praegune teenus ei näinud, ja
kaotab 88, mida teenus nägi. **Netos võidab, aga see on vahetus, mitte
ülekaal** — ja kumb neist 225-st reast on õige, seda ei saa kummagi väljundi
põhjal otsustada. Selleks on vaja päris ground truth'i.

Nimekiri lehtedest, kus nad kõige rohkem lahku lähevad, koos piltide ja
mõlema teksti teedega: `data/vutt/reocr/markup-Q8_0-png/SILMAGA-VAADATA.md`.

---

## Mis on lahtine

1. **Kaks lehte kaheksast** (`0017`, `0036`) kaotavad marginaalid ka lõpliku
   ahelaga. Kvantimine on välistatud (BF16 = Q8_0), pildiahel on välistatud.
   Järgmine samm oleks visuaalenkoodri vahe-embeddingute otsevõrdlus
   transformersi ja llama.cpp vahel, mitte enam samplerid ega pilditöötlus.
2. **`<cs>` jääb poole peale** (131 → 60) ja seda ei muutnud ükski pildiparandus.
   Seni uurimata.
3. **Kumb pool marginaalides õigem on** — vajab silmaga kontrolli, vt nimekiri.
4. **Ülesvoolu PR** llama.cpp-le: konverter peaks kirjutama
   `clip.vision.image_max_pixels` võtme mudeli enda konfist.

---

## Mida sellest kaasa võtta

**Vaikeväärtus, mida keegi ei näita, on ohtlikum kui viga.** 4096-tokeni piir ei
tekita hoiatust, ei jäta logisse rida ega muuda väljundit ilmselgelt katkiseks.
Leht tuleb tagasi tervena — lihtsalt ühe veeru võrra vaesemana.

**Mõõtmisahel ise on osa katsest.** Kahest leitud põhjusest **üks oli minu enda
kliendi oma**, mitte mudeli või mootori oma. Ilma selleta oleks järeldus olnud
„llama.cpp visuaalahel ei näe ääreveergu" — usutav, vale ja pikaks ajaks
kalliks minev.

**Koondkeskmine varjab täpselt seda, mida vaja näha.** CER 13,9 % vs 15,6 %,
marginaalide 179 vs 0 — mõlemad tulevad üksikutest lehtedest. Iga järeldus siin
tuli lehe tasandil vaatamisest, mitte tabeli alumiselt realt.

**Võrdlusalus ei ole tõde.** Praegune tootmisväljund on lihtsalt teine arvamus,
ja nagu selgus, ise ka treeningust nihkes. Iga „X on halvem kui Y" järeldus
kehtib ainult niikaua, kuni Y on tõestatult õige.

---

## Parandused v1 suhtes

| v1 väitis | tegelikult |
|---|---|
| „17 % lehti jooksis 4096 tokeni lakke" | **6 %** (8/143), ja see on **väljundi** lagi, mitte sisendi visuaaltokenite oma. 17 % oli jooksu esimese 23 päringu põhjal. |
| „marginaalide 203 vs 0" | 203 sisaldas loopinud lehte; õiged arvud on **166** (VUTT) või **179** (unsloth). |
| „kaks põhjust kolmest olid minu kliendi omad" | Põhjusi on **kaks**, neist **üks** on minu kliendi oma. |
| „visuaaltokeneid täpselt mõõta" | Prompt-tokenite vahe sisaldab ka wrapper-tokeneid; täpseks kontrolliks on vaja ka valemit `(H/32)×(W/32)`. |
| Q8_0 välistatud (JPEG-ahelaga) | Korratud **lõpliku ahelaga** — järeldus jäi samaks, aga tõend on nüüd kehtiv. |
| Valehäirete kontroll 419 „terve" peal | Korratud **kõigi 430 mitte-loobi** peal, sh 11 halli. Lisaks leitud 1 valehäire 1189 inimese kinnitatud teksti peal. |
| „ükski konverter ei kirjuta pikslieelarvet" | Kehtib, ja **kontrollitud ka Unslothi ametliku GGUF-i peal** — võtmed puuduvad ka seal. |

---

*Andmed: `data/kurrent/eval/` ja `data/vutt/reocr/`, iga jooksu tingimused
`run.json`-is. Skriptid: `eval_kurrent.py`, `reocr_vutt.py`, `loop_detect.py`,
`imaging.py`, `merge_lora.py`.*
