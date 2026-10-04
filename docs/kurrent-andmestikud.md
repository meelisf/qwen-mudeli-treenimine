# Kurrent OCR treeninguandmestikud

**Viimati uuendatud:** 2026-07-08 (allpool olev nimekiri kirjeldab 2026-06-03 seisu)
**Kogumahu** (data/kurrent/metadata.csv, tollal): **16 579 lehekülge**

**Treeningu seis:** hetkel aktiivne käsikirjaline mudel on `models/qwen3.5-ocr-kurrent-20260602`
(checkpoint kirjutatud 03.06.2026 kell 17:34). Allpool loetletud 13 allikat vastavad
täpselt sellele, mis treeningusse läks. `data/kurrent/metadata.csv` muudeti aga
**04.06.2026 kell 11:57** — pärast treeningu lõppu — kui lisati kaks uut allikat:
**dresdner_1665** (241 lk, TEI XML, `build_dresdner_tei_dataset.py`) ja
**senatsprotokolle** (229 lk, `build_senatsprotokolle_dataset.py`). Need kaks
**EI OLE** praeguses mudelis kasutatud, vaid ootavad järgmist treeningvooru.
(Vt ka eraldiseisev, väiksem `dresdner_hofdiarium_1673`, 20 lk, allpool #13 —
see OLI treeningus sees.)

Kõik andmestikud on töödeldud `data/kurrent/` formaati: JPEG pildid + `metadata.csv` (veergud: `failinimi`, `transkriptsioon`, `allikas`).

---

## Tööriistad

| Skript | Eesmärk |
|--------|---------|
| `scripts/filter_dataset.py --stats` | Vaata allikate jaotust |
| `scripts/filter_dataset.py --max-per-source 1000 --out data/kurrent/metadata_balanced.csv` | Tasakaalustatud treening-CSV |
| `scripts/filter_dataset.py --exclude kurrent_xix --out ...` | Jäta allikas välja |
| `scripts/add_allikas_column.py` | Üks kord: lisab `allikas` veeru olemasolevale CSV-le |

---

## Andmestikud

### 1. kurrent_xix — 8 000 lk
**Allikas:** [dh-unibe/image-text_kurrent-xix](https://huggingface.co/datasets/dh-unibe/image-text_kurrent-xix)  
**Keel:** Saksa  
**Periood:** XIX sajand  
**Stiil:** Saksa Kurrent, XIX sajandi kantseleikirjutus  
**Formaat:** Parquet (PIL pildid + PAGE XML string)  
**Litsents:** CC BY 4.0  
**Skript:** `scripts/build_kurrent_dataset.py`  
**Märkus:** 33 projekti, max 250 lk/projekt. Domineerib praeguses andmestikus (48%).

---

### 2. aaeb_xiv_xvii — 1 992 lk
**Allikas:** [dh-unibe/image-text_aaeb-xiv-xvii](https://huggingface.co/datasets/dh-unibe/image-text_aaeb-xiv-xvii)  
**Keel:** Saksa  
**Periood:** XIV–XVII sajand  
**Stiil:** Varane Kurrent, gooti kirjutus  
**Formaat:** Parquet  
**Litsents:** CC BY 4.0  
**Skript:** `scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_aaeb-xiv-xvii`

---

### 3. bullinger_autoren — 1 837 lk
**Allikas:** [dh-unibe/image-text_bullinger-autoren](https://huggingface.co/datasets/dh-unibe/image-text_bullinger-autoren)  
**Keel:** Saksa / ladina  
**Periood:** XVI sajand  
**Stiil:** Humanistlik kirjutus, Kurrent, Bullinger-Briefwechsel  
**Formaat:** Parquet  
**Litsents:** CC BY 4.0  
**Skript:** `scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_bullinger-autoren`

---

### 4. bergskollegium_rel_seg — 1 439 lk
**Allikas:** [Riksarkivet/bergskollegium_relationer_och_skrivelser_seg](https://huggingface.co/datasets/Riksarkivet/bergskollegium_relationer_och_skrivelser_seg)  
**Keel:** Rootsi / saksa  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, kaevanduskolleegium dokumendid  
**Formaat:** tar.gz (JPG pildid + PAGE XML)  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset bergskollegium_relationer_och_skrivelser_seg`

---

### 5. hanse_kurrent_xvi — 1 144 lk
**Allikas:** [fgho/hanse-kurrent-xvi-rawxml](https://huggingface.co/datasets/fgho/hanse-kurrent-xvi-rawxml)  
**Keel:** Saksa  
**Periood:** 1505–1595 (XVI sajand)  
**Stiil:** Hansaliidu Kurrent (Lübeck, Stralsund, Köln, Hamburg jt)  
**Formaat:** Parquet (image dict + PAGE XML string)  
**Litsents:** MIT  
**Skript:** `scripts/build_hanse_dataset.py`  
**Märkus:** Varaseima perioodi saksa Kurrent andmestikus.

---

### 6. svea_hovratt_seg — 847 lk
**Allikas:** [Riksarkivet/svea_hovratt_seg](https://huggingface.co/datasets/Riksarkivet/svea_hovratt_seg)  
**Keel:** Rootsi  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, Svea hovrätt kohtudokumendid  
**Formaat:** tar.gz (JPG + PAGE XML); raw failid: `data/raw/svea_hovratt_seg_*.tar.gz`  
**Skript:** `scripts/build_svea_dataset.py` (lokaalsetest failidest) või `scripts/build_riksarkivet_dataset.py`

---

### 7. trolldomskommissionen_seg — 761 lk
**Allikas:** [Riksarkivet/trolldomskommissionen_seg](https://huggingface.co/datasets/Riksarkivet/trolldomskommissionen_seg)  
**Keel:** Rootsi  
**Periood:** XVII sajand  
**Stiil:** Rootsi kantseleikirjutus, nõiaprotsesside dokumendid  
**Formaat:** tar.gz (JPG + PAGE XML)  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset trolldomskommissionen_seg`

---

### 8. krigshovrattens_seg — 343 lk
**Allikas:** [Riksarkivet/krigshovrattens_dombocker_seg](https://huggingface.co/datasets/Riksarkivet/krigshovrattens_dombocker_seg)  
**Keel:** Rootsi  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, sõjakohtu doomiraamatud  
**Formaat:** tar.gz (JPG + PAGE XML)  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset krigshovrattens_dombocker_seg`

---

### 9. jonkopings_seg — ~57 lk (sh ~18 duplikaati)
**Allikas:** [Riksarkivet/jonkopings_radhusratts_och_magistrat_seg](https://huggingface.co/datasets/Riksarkivet/jonkopings_radhusratts_och_magistrat_seg)  
**Keel:** Rootsi  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, raekohtu ja magistraadi dokumendid  
**Formaat:** tar.gz (JPG + PAGE XML); raw failid: `data/raw/jonkopings_*.tar.gz`  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset jonkopings_radhusratts_och_magistrat_seg`  
**Märkus:** ~18 duplikaati skripti kahekordsest käivitamisest – tühine mõju.

---

### 10. bergskollegium_adv_seg — 53 lk
**Allikas:** [Riksarkivet/bergskollegium_advokatfiskalskontoret_seg](https://huggingface.co/datasets/Riksarkivet/bergskollegium_advokatfiskalskontoret_seg)  
**Keel:** Rootsi / saksa  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, advokaadifiskaali kantselei  
**Formaat:** tar.gz (JPG + PAGE XML)  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset bergskollegium_advokatfiskalskontoret_seg`

---

### 11. gota_hovratt_seg — 51 lk
**Allikas:** [Riksarkivet/gota_hovratt_seg](https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg)  
**Keel:** Rootsi  
**Periood:** XVII–XVIII sajand  
**Stiil:** Rootsi kantseleikirjutus, Göta hovrätt dokumendid  
**Formaat:** tar.gz (JPG + PAGE XML)  
**Skript:** `scripts/build_riksarkivet_dataset.py --dataset gota_hovratt_seg`

---

### 12. koenigsfelden_adhr — 34 lk
**Allikas:** [dh-unibe/image-text_koenigsfelden-adhr-colmar](https://huggingface.co/datasets/dh-unibe/image-text_koenigsfelden-adhr-colmar)  
**Keel:** Saksa  
**Periood:** XIX sajand  
**Stiil:** Saksa Kurrent  
**Formaat:** Parquet  
**Litsents:** CC BY 4.0  
**Skript:** `scripts/build_kurrent_dataset.py --dataset dh-unibe/image-text_koenigsfelden-adhr-colmar`

---

### 13. dresdner_hofdiarium_1673 — 20 lk
**Allikas:** [Zenodo 10.5281/zenodo.15303243](https://zenodo.org/records/15303243)  
**Keel:** Saksa  
**Periood:** 1673 (XVII sajand)  
**Stiil:** Saksoni Kanzleikurrent, Dresdner Hofdiarium (SLUB Mscr.Dresd.K.117)  
**Formaat:** JPG + ALTO XML v4 (Zenodost otse, eraldi failidena)  
**Litsents:** CC BY-NC-SA 4.0  
**Skript:** `scripts/build_dresdner_dataset.py`  
**Märkus:** Väike aga kvaliteetne; ainus ALTO XML formaadis andmestik meil.

---

## Ülevaatus 2026-10-02: mis on uut ja mis jäi kasutamata

Taust: VUTT-i võrdlus (VUTT repo `docs/reviews/2026-10-02-htr-mudelite-vordlus.md`)
näitas, et `kurrent-20260829` loeb selget kätt Gemini 3.8 Flashiga samal
tasemel, aga isiklikke ja õpetlaste käsi (nt Morgensterni XIX saj algus)
oluliselt kehvemini.

### kurrent_xix = AINULT Zürichi valitsusprotokollid

`build_kurrent_dataset.py` voogedastab andmestikku shard'ide järjekorras ja
lõpetab 8000 lehe juures (250 lk/projekt). Esimesed projektid on `MM_1_001…`:
**kõik meie 8000 kurrent_xix lehte on 33 köidet Zürichi Regierungsratsprotokolle
(1803–1883)** — üks arhiivisari, puhtand, kantseleikäsi, 47 % kogu korpusest.
Kontrollitud: failinimede eesliited `083605…083661` = `MM_1_001…MM_1_033`
(HF rows-API: offset 0 = `MM_1_001`, `083605_…`).

Andmestikus on praegu 321 projekti / 144 533 lk (muudetud viimati 2026-04-25,
ainult duplikaat-shard'ide kustutamine). Kasutamata on 121 mitte-Zürichi
projekti (~78 GB): READ/CITlab Kurrendi GT (`TRAIN_CITlab_*`, `TEST_CITlab_*`,
`TRAINING_TESTSET_*`) + `hufeland_privatbesitz_1829`, `nn_msgermqu2124_1827`,
`nn_msgermqu2345_1827`, `parthey`, `semper_20_MS`.

Pisteline kontroll (alla laaditud `~/.cache/huggingface/hub/datasets--dh-unibe--image-text_kurrent-xix`):

| projekt | lk | read/lk | märkus |
|---|---|---|---|
| `TRAIN_CITlab_GrimmBriefe` | 7 | 20 | 1830–40ndad, Grimmide kirjad — õige tüüp, aga tilluke |
| `TRAIN_CITlab_Tagebuch_Arnold_v1` | 89 | 20 | **1940ndad** (II maailmasõda), mitte XIX saj |
| `TRAIN_CITlab_Minutes_of_Estonian_Knighthood` | 130 | 60 | **1900–1915**, kalligraafiline puhtand, kahelehelised laotused |
| `nn_msgermqu2124_1827` | 175 | mediaan 0 | 115/175 lehte < 5 rea — **valdavalt transkribeerimata** |
| `parthey` | 801 | mediaan 0 | 523/801 lehte < 5 rea — **valdavalt transkribeerimata** |

Järeldused:
- Nimi „xix" ei taga perioodi — iga projekt tuleb enne kasutamist dateerida.
- Paljud projektid on duplikaadid: `TRAIN_`/`TEST_`/`TRAINING_TESTSET_` on sama
  projekti eri jaotused; `Lotte1` = `Lotte_Tagebuecher` = `Lottes_Schrift`
  (kõik 613 MB), `Haeckermann_3+` = `Konzilsprotokolle_A_Haeckermann_3`,
  kaks `Protokoll_Hoftheater_1806-Vers_2018Nov` varianti. Dedup sisu järgi.
- Tühjad lehed (< `MIN_LINES`) filtreerib build-skript juba välja.
- Järgmisel ehitusel **piira Zürichi osakaalu ja võta mitte-MM projektid
  eraldi** — praegune voogedastus ei jõua nendeni kunagi.

### kurrent_xix mitte-Zürichi audit (2026-10-02, täielik)

Alla laaditud kõik 121 mitte-`MM_` projekti (73 GB, HF cache). Skript
`scripts/xix_audit.py` → `data/kurrent_xix_audit/`:
`pages.csv` (lehe kaupa: read, aastad, teksti- ja pildiräsi), `projects.csv`,
`dups.txt` (projektipaarid jagatud lehtedega), `samples/` (2 pisipilti/projekt),
**`unique_pages.csv` = dedup'itud, transkribeeritud lehtede kanooniline loend**.

**Dedup:** union-find kolme võtmega — teksti räsi (normaliseeritud, ≥ 5 rida),
pildi räsi, Transkribuse pageId (failinime viimane osa). Esindajaks `TRAIN_`
eelistatult, siis kõige rohkem ridu.

| | lehti |
|---|---|
| kõik read 121 projektis | 30 676 |
| unikaalseid lehti | 22 483 |
| **unikaalseid transkribeeritud (≥ 5 rida)** | **11 249** |

Duplikaate on massiliselt: `Haeckermann_3+` = `Konzilsprotokolle_A_Haeckermann_3`
(1 921 ühist), mõlemad ⊃ `Konzilsprotokolle_M4_HTR+` (1 655), `Todesurteile`
kaks varianti (768), `Konzilsprotkolle_B_Schwartz` ≈ `Schwartz_M6` (526),
Hoftheater variandid, Nekrolog ×4, Pyl M1–M5 kattuvad ahelana. `TEST_` ja
`TRAINING_TESTSET_` on enamasti `TRAIN_`-i alamhulgad.

**Perioodid** (projekti aasta nimest, muidu tekstis mainitud aastate mediaan;
`parthey` ja `semper_20_MS` parandatud käsitsi — vt allpool):

| periood | lk | peamised |
|---|---|---|
| **1750–99** | **2 566** | Haeckermann/Konzilsprotokolle 1 921 + 87 (Greifswaldi ülikooli konsiilium 1775–1811), Schwartz 542 + 14 (1755–86) |
| 1800–49 | 2 545 | Todesurteile 1849– 803, OEAW 460 (1847–49), Hoftheater 1806 286, Pyl 393, parthey 278, Müller 97, hufeland 52, nn_msgermqu 84 |
| 1850–99 | 4 606 | **Escher 3 819** (kirjad 1843–77), semper 260, Todesurteile 237, Gusbeth 114, Bassermann 106 |
| 1900+ | 1 429 | Roland 1941, Steiner, Nekrolog, Estonian Knighthood 1905–15, Kochbuch 1930ndad, Arnold 1940 |
| ? | 103 | Barlaam, Suppes, MargareteSick |

**Peamine leid: 18. sajandi saksa Kurrenti avalik GT ON olemas** — ~2 500 lehte
ülikooli konsiiliumi protokolle (Greifswald 1775–1811, Schwartz 1755–86),
täpselt Tartu ülikooli materjali žanr. Varasem väide („XVIII saj saksa
Kurrenti avalikke treeningandmeid ei eksisteeri", `docs/arhiiv/kurrent-strateegia.md`)
**ei kehti** — andmed olid kurrent_xix-is, aga voogedastus ei jõudnud nendeni.

Silmaga kontrollitud (näidised `samples/`):
- `Haeckermann_3+`: Greifswald, 5. aprill 1796, rektori ja professorite
  protokoll — akadeemiline kantseleikurrent, mitu kätt.
- `parthey`: Humboldti „Kosmose" loengute konspekt (Gay-Lussac, Quito) ≈ 1827–28,
  **ladina kirjas, mitte Kurrendis**. Tekstis mainitud aastad (mediaan 1783)
  eksitavad → periood käsitsi 1800–49.
- `semper_20_MS`: Gottfried Semper, 1850ndad; tekstiaastate mediaan 1761
  eksitav → käsitsi 1850–99.
- `Escher_M1`: Alfred Escheri kirjad (nt 21.08.1867), kiire isiklik käsi —
  just see mitmekesisus, mis Zürichi puhtandist puudub. NB: 13 538 lehest
  ainult 3 819 transkribeeritud.

**Järeldused järgmiseks Kurrendi treeninguks:**
- Uut 1750–1899 materjali on ~9 700 unikaalset lehte, eri kätest. See on
  suurem panus kui kogu hanse-xvii, ja just mitmekesisuse poolest.
- `Escher` (3 819) ja Zürich (8 000) kalduvad domineerima — piira
  projekti kohta (nt 500–1000).
- 1900+ (1 429) võib välja jätta või hoida väikesena — VUTT-il seda materjali
  peaaegu ei ole.
- Ehitusskript peab lugema `unique_pages.csv`-d (projekt + failinimi), mitte
  voogedastama andmestikku järjekorras.

### Ehitatud: andmestik v2 (2026-10-02)

`scripts/build_kurrent_v2.py` → `data/kurrent/` (vana kaust prügikastis).
Zürich ≤ 1 000 (holdout-lehed alati sees), Escher ≤ 1 000, 1900+ ≤ 50/projekt,
ülejäänud `unique_pages.csv` lehed kõik, hanse-xvii kõik; MIN_LINES nagu vanadel
skriptidel (xix 5, hanse 3), tekst samadest `parse_pagexml`-idest. Seed 3407.
Kontrollitud: 18 908 unikaalset faili, 0 vigast pilti, 0 orbu, 0 tühja teksti,
133 holdout-rida kõik CSV-s. Hõredaid (≤ 200 märki) 581 — VUTT-i 27
tühja/hõredat lehte piisab, hõreduse äratundmine on lihtne õppida.

### fgho/hanse-kurrent-xvii-rawxml — UUS (2026-07-09), alla laaditud

`~/.cache/huggingface/hub/datasets--fgho--hanse-kurrent-xvii-rawxml` (10,8 GB,
35 parquet-faili). **1 298 lk**, neist 12 < 5 rea. Dekaadid: 1600ndad 277,
1610ndad 629, 1620ndad 272, 1660ndad 120. Alamsaksa linnapäevade retsessid
(Lübecki, Kölni, Braunschweigi jt arhiivid), 19–44 rida lehel. Poolitus `¬`
(sama mis `KURRENT_INSTRUCTION`). Ladina sõnad antiikvas keset Kurrenti
(„Recesses", „deliberation", „contributionibus") — sama code-switching nagu
VUTT-i materjalis. Sama kuju kui `hanse_kurrent_xvi`, seega `build_hanse_dataset.py`
peaks sobima (andmestiku nimi on skriptis kõvakoodis, rida ~115).
NB: mitu retsessi on sama koosoleku eri arhiivide koopiad (nt 1669 viies
eksemplaris) — eri käed, sama tekst; pigem pluss kui duplikaat.

### Teised kandidaadid

- Zenodo 15303398 Dresdner Hofdiarium 1653–56: 12 lk, CC BY 4.0, ei ole veel
  kasutusel (meil 1665 ja 1673). `build_dresdner_dataset.py`.
- `fgho/hanse-kurrent-xv` (429 lk, XV saj) ja dh-unibe XIV–XVI saj kogud —
  liiga vara.
- Riksarkivet `goteborgs_poliskammare_fore_1900`, `frihetstidens_utskottshandlingar`
  (mitte-`_seg`): laadimisskriptiga, maht teadmata — kontrollida, kas kannab teksti.
- XVIII saj saksa Kurrent: vt kurrent_xix audit ülal — ~2 500 lehte
  ülikooli konsiiliumi protokolle (Greifswald, Schwartz). VUTT-i oma „Valmis"
  käsikirjamaterjal jääb siiski ainsaks Baltikumi-spetsiifiliseks allikaks.

## Ülevaatus 2026-10-04: osaline GT ja andmestik v4

Taust: GT kiirkontroll (SEIS §6.11 tee c, `scripts/gt_kontroll.py`, mudel
`kurrent-20261002` üle treeningkomplekti) näitas kandidaatide koondumist kahte
allikasse: Escher (50 % lehtedest) ja `bullinger_autoren` (30 %). Silmaga
kontrollitud: GT-st puuduvad read, mis pildil on olemas (nt Escher 4505: 7 rida
20-st), ja mudel on ridade vahelejätmist juba õppinud (17717, 7005).

### Põhjus: tühjad TextLine'id jäetakse vaikselt vahele

`build_kurrent_dataset.parse_pagexml` jätab teksti(ta) TextLine'i vahele
(`if uc is None or not uc.text: continue`). Pilt näitab rida, GT-s seda pole.

Kust tühjad read tulevad:
- **CITlabi „Matcher"** (`<Creator>`-is `Matcher(net_0.sprnn…)`): editsioonitekst
  on automaatselt joondatud tuvastatud ridadele; kus joondus ebaõnnestus, jäi
  rida tühjaks. Matcher-projektid: Escher, `semper_20_MS`, `Pyl_M3–M5`,
  `parthey`, `hufeland_privatbesitz_1829`, `nn_msgermqu2124/2345`.
- **Bullinger**: 80 % treeningulehtedest ≥ 2 tühja rida (Transkribus);
  lisaks on **sama fail HF-andmestikus mitmes versioonis eri XML-iga** —
  voogedastus võttis esimese, 100 lehel oli täielikum versioon olemas.
- AAEB, Hanse XVI/XVII, Königsfelden: praktiliselt puhtad.

Mõõdik: `scripts/tuhjad_read_audit.py` → `data/kurrent_xix_audit/tuhjad_read.csv`
(xix + hanse, kohalik HF cache) ja `--kaug` → `tuhjad_read_kaug.csv`
(Bullinger, AAEB, Königsfelden otse HF-ist, ainult `xml_content` veerg —
pilte ei laadita). Veerud: allikas, projekt, failinimi, ridu, tühje, looja,
`treeningus` (andmestiku failinimi).

**Lävi on mõõdetud, mitte valitud.** GT-kontrolli kandidaatide osakaal
(normaliseeritud CER ≥ 10 % mudelil, mis on neid lehti treeningus näinud):

| tühje TextLine'e | xix: kandidaate | Escher | Bullinger (õige versioon) |
|---|---|---|---|
| 0 | 5 % | 1/8 | 11 % (0–1 kokku) |
| 1 | 3 % | 0/13 | |
| ≥ 2 | **58 %** | **62 %** | **35 %** |

Üksik tühi rida on enamasti müra (tempel, kriips) → lävi **≥ 2 tühja rida = välja**.

### DTA Kosmos-Nachschriften: osalise GT asemel täistekst

Matcheri allikas on Deutsches Textarchiv (DTA). DTA TEI kannab sama käsikirja
**täielikku lehe teksti**; treenime lehe tasemel, nii et reajoondust ei ole
vaja. `<pb facs="#fNNNN">` = xix failinime lehenumber (parthey 801 = 801,
hufeland 172, nn_msgermqu 175/341 — kontrollitud silmaga parthey 673).

Loend: HU Berlin „Nachschriften der Kosmos-Vorträge" (Christian Thomas).
Kasutusel 9 (CC BY 4.0): `parthey_msgermqu1711_1828`, `hufeland_privatbesitz_1829`,
`nn_msgermqu2124_1827`, `nn_msgermqu2345_1827` (olid xix-is Matcheriga) +
**5 uut kätt**: `libelt_hs6623ii_1828`, `patzig_msgermfol841842_1828`,
`willisen_humboldt_1827`, `nn_oktavgfeo79_1828`, `nn_n0171w1_1828`.
`riess_f2e1853_1828` on DTAQ-s (kvaliteedikontrollis), avalikult ei saa.
Lohde ja Stenmark ei ole DTA-s (ainult pildid).

Allalaadimine: TEI `https://www.deutschestextarchiv.de/book/download_xml/<id>`
(nõuab küpsist `verified=1`), pildid
`https://media.dwds.de/dta/images/<id>/<id>_<NNNN>_1600px.jpg`.
Vahemälu `data/dta_tei/` (TEI + `img/`).

TEI → diplomaatiline tekst (`scripts/build_dta_kosmos.py`):
- `choice` → `abbr` | `orig` | `sic` (MITTE `expan`/`reg`/`corr` — muidu „u̅" → „uund");
  üksik `expan`/`reg`/`corr`/`supplied` välja
- `del rendition="#s"` (läbikriipsutus) jääb, `#ow`/`#erased` (ülekirjutatud) välja
- `add`, `unclear`, `hi`, `fw`, marginaal-`note` jäävad; `note type="editorial"` välja;
  `metamark` välja
- `ſ` → `s` (ülejäänud Kurrendi GT-s ſ-i ei ole); rea lõpu `-` jääb nagu DTA-s
  (korpus on niikuinii segi: `¬` ja `-`)
- **leht välja**, kui seal on `gap` (loetamatu/kadunud: 246), tabel (21),
  U+FFFC (esitamatu lühendusmärk: 656, enim Patzig) või < 5 rida (140)

Tulemus: 3 526 lehest **2 463**. Proovida `--naita <id> <nr>`.

### Ehitatud: andmestik v4 (2026-10-04)

Neli sammu, iga samm oma kausta (pildid hardlink'itud, sisend puutumata):

| samm | skript | kaust | lehti |
|---|---|---|---|
| v2 miinus ≥ 2 tühja reaga xix-lehed | `build_kurrent_v3.py` | `data/kurrent_v3/` | 18 908 → 17 224 (−1 684) |
| Bullinger uuesti: parim XML-versioon, ≤ 1 tühi rida | `build_bullinger_v3.py` | `data/bullinger_v3/` | 707 (540 pilti olemas, 169 HF-ist) |
| DTA Kosmos | `build_dta_kosmos.py` | `data/dta_kosmos/` | 2 463 |
| Escheri kõik puhtad unikaalsed lehed | `build_escher_lisa.py` | `data/escher_lisa/` | +567 |
| koondamine | `build_kurrent_v4.py` | **`data/kurrent_v4/`** | **18 983** |

v3-st eemaldatud projekti kaupa: Escher −799, semper −259 (≈ kõik), parthey −184,
Pyl M3–M5 −313, nn_msgermqu −57, hufeland −23, ülejäänud üksikud.
v4 koondamisel: vana Bullinger −1 828 (holdout'i 9 jäävad), parthey/hufeland/
nn_msgermqu Matcheri jäägid −150 (asendab DTA), + Bullinger v3, DTA, Escher.

Escher: kasutaja vaatas pildid üle (04.10) — eri käed, VUTT-ile lähedased;
v2 Escheri lagi 1 000 kaotati. Unikaalseid transkribeeritud lehti 3 819, neist
puhtaid (≤ 1 tühi, ≥ 5 rida) 768 — rohkem puhast Escheri ei ole (ülejäänud
~9 700 toorkirjet on transkribeerimata lehed).

Allikad v4-s (lehti): xix_read_1750_99 2 564, **dta_kosmos_1827 2 463**,
aaeb 1 982, xix_read_1800_49 1 793, bergskollegium_rel 1 439, hanse_xvii 1 292,
xix_read_1850_99 1 287, hanse_xvi 1 144, zurich 1 000, svea 847, trolldom 761,
**bullinger 716**, xix_read_1900 559, krigshovratt 343, dresdner 241, senats 229,
dateerimata 101, jonkopings 57, bergskollegium_adv 53, gota 51, koenigsfelden 34,
vutt_horedad 27. Periood 1800–49 kokku ~4 260 (v2: ~1 940).

**Holdout 133 → 128**: 5 lehte olid sama vea all (Pyl, Escher ×2, semper, Dreier)
ja on eemaldatud ka treeningust. Võrdle mudeleid nende 128 peal.

Kontrollitud: 18 983 unikaalset failinime, 0 puuduvat pilti, 0 tühja teksti,
128/128 holdout-rida CSV-s.

### Tootmismudel ja ridade vahelejätmine

Ridade vahelejätmise võis tuua alles v2 (29.08 andmestikus Matcheri lehti ei
olnud). Mõõdetud eval-väljunditest (GT rida ≥ 8 märki, mille parim vaste
väljundis < 0,6): vanad 73 holdout-lehte — `kurrent-20260829` 29 puuduvat rida
(15 lehel), `kurrent-20261002` 31 (13 lehel) = **viik, tagasivahetust ei ole
vaja**. Kõigil 128-l: 177 → 48. Halvim üksikjuhtum 22229 (uus jätab osa lehe
igast teisest reast vahele).

### Enne treeningut veel lahti

- GT-kontroll (jookseb `data/kurrent`-i peal, lõpp ~05.10 17:00): v2-st pärit
  lehtedel CER ≥ 10 % = mudel nägi lehte treeningus → tugev GT-vea märk.
  Vaata iga allika kohta näiteid ENNE hulgi väljaviskamist.
- Uued lehed (DTA, Escher-lisa, Bullingeri uued versioonid) läbi sama mudeli,
  aga sõelu AINULT struktuursete tunnuste järgi (väljund GT-st selgelt pikem,
  loop, GT algab keset lehte) — mudel ei ole neid käsi näinud, kõrge CER ei
  tõenda GT viga.
- Vahetus: `data/kurrent_v4` → `data/kurrent` (praegune kaust on GT-kontrolli
  sisend — mitte enne selle lõppu). `train_kurrent.py` loeb `data/kurrent/`.
- Toor-XML (`~/_kustutamiseks_20261002/hf_cache/`) **ära kustuta enne uut
  treeningut** — kõik ehitusskriptid loevad sealt.

---

## Vaadatud aga mitte kasutusel

| Andmestik | Põhjus |
|-----------|--------|
| [Riksarkivet/frihetstidens_utskottshandlingar_seg](https://huggingface.co/datasets/Riksarkivet/frihetstidens_utskottshandlingar_seg) | Tühjad transkriptsioonid – segmenteeritud aga transkribeerimata |
| [Zenodo 10.5281/zenodo.17252677](https://zenodo.org/records/17252677) – German Kurrent HTR 9317 rida | Reataseme andmestik (lõigatud read), mitte täisleheküljed; kattub kurrent_xix-ga. **04.10:** sama DTA materjal on v4-s lehe tasemel otse DTA TEI-st (vt „Ülevaatus 2026-10-04“) |
| [Zenodo 10.5281/zenodo.19728926](https://zenodo.org/records/19728926) – BullingerDB 20 898 lk / 376 582 rida | **Binariseeritud** (must-valge) pildid – ei sobi värvilistel skaneeringatel treenitud mudelile; sama Bullinger XVI saj sisu mis meil juba `bullinger_autoren`-is; 99.8 GB allalaadimine; reataseme rekonstrueerimine keeruline. Vt artikkel: arxiv.org/abs/2605.30235 |
| [aarhus-city-archives/historical-danish-handwriting](https://huggingface.co/datasets/aarhus-city-archives/historical-danish-handwriting) – >11 000 lk, 15.2 GB, CC-BY-4.0 | Taani käsikiri 1841–1939. Tähekujud identsed saksa Kurrentiga, aga: (1) enamus lehekülgi on **ladina kirjas** (Taani loobus Kurrentist ~1875–1885, andmestik ulatub 1939-ni); (2) taani sõnavara treeninguandmetes segab mudeli keelemudelit – võib saksa/rootsi OCR-i halvemaks teha. Kasutatav ainult kui Kurrent-periood (1841–1880) oleks eraldatav. |

---

## Perioodide ja stiilide kaart

```
XVI saj    hanse_kurrent_xvi (Hansaliit, saksa Kurrent)
           bullinger_autoren (Šveitsi, saksa/ladina humanistlik)
           aaeb_xiv_xvii (lõpuosa)

XVII saj   aaeb_xiv_xvii (algusosa, kuni ~1650)
           dresdner_hofdiarium_1673 (Sakson, Kanzleikurrent)
           trolldomskommissionen_seg (Rootsi kantselei)
           krigshovrattens_seg (Rootsi kantselei)
           svea_hovratt_seg (Rootsi kantselei)
           bergskollegium_*_seg (Rootsi/saksa kantselei)
           jonkopings_seg (Rootsi kantselei)
           gota_hovratt_seg (Rootsi kantselei)

XVIII saj  svea_hovratt_seg, krigshovrattens_seg jt (Rootsi, jätkuvad)

XIX saj    kurrent_xix (Šveits/Saksamaa, 8 000 lk – suurim)
           koenigsfelden_adhr (34 lk)
```
