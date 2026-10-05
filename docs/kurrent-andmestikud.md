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
20-st), ja mudel on ridade vahelejätmist õppinud (17717, 7005).

### Põhjus: tühjad TextLine'id jäetakse vaikselt vahele

Kõik PAGE XML-i ehitajad (`build_kurrent_dataset`, `build_riksarkivet_dataset`,
`build_senatsprotokolle_dataset` jt) jätavad teksti(ta) TextLine'i vahele
(`if uc is None or not uc.text: continue`). Pilt näitab rida, GT-s seda pole.

Kust tühjad read tulevad:
- **CITlabi „Matcher"** (`<Creator>`-is `Matcher(net_0.sprnn…)`): editsioonitekst
  on automaatselt joondatud tuvastatud ridadele; kus joondus ebaõnnestus, jäi
  rida tühjaks. Matcher-projektid: **Zürich (kõik)**, Escher, `semper_20_MS`,
  `Pyl_M3–M5`, `parthey`, `hufeland_privatbesitz_1829`, `nn_msgermqu2124/2345`.
  Täidetud read võivad olla ka VALED (Zürich lk 308: „56." Transpordi real,
  lehenumber 308 → „200.").
- **Bullinger**: 80 % treeningulehtedest ≥ 2 tühja rida (Transkribus); lisaks on
  **sama fail HF-andmestikus mitmes versioonis eri XML-iga** — voogedastus
  võttis esimese, 100 lehel oli täielikum versioon olemas.

**GT-kontroll üksi seda viga EI püüa.** Zürichi mediaan-CER oli 0,5 % ja
kandidaate 0: mudel on 8 000 Zürichi lehelt õppinud samu ridu vahele jätma,
väljund ja GT jätavad sama vahele. Seepärast auditeeritakse KÕIK allikad XML-i
tasemel (allpool), GT-kontroll on teine kiht.

**Lävi on mõõdetud, mitte valitud.** GT-kontrolli kandidaatide osakaal
(normaliseeritud CER ≥ 10 % mudelil, mis on neid lehti treeningus näinud):

| tühje TextLine'e | xix: kandidaate | Escher | Bullinger (õige versioon) |
|---|---|---|---|
| 0 | 5 % | 1/8 | 11 % (0–1 kokku) |
| 1 | 3 % | 0/13 | |
| ≥ 2 | **58 %** | **62 %** | **35 %** |

Üksik tühi rida on enamasti müra (tempel, kriips) → lävi **≥ 2 tühja rida = välja**
(ka holdout'ist: katkine GT ei sobi ka mõõtmiseks).

### Audit: kõik 24 allikat

Tühjade ridade mõõtmine, väljundid `data/kurrent_xix_audit/tuhjad_read*.csv`
(veerud: ridu, tühje, `treeningus` = andmestiku failinimi). Kus failinimi on
lõigatud (Riksarkiv: 60 märki, lehe ID kaob), seotakse treeningrida XML-iga
**teksti järgi** (sama `parse_pagexml`). Mitme XML-versiooni korral valitakse
see, mille täidetud ridade arv = GT ridade arv.

| allikas | skript / kiht | tulemus v4-s |
|---|---|---|
| xix (5 perioodi) | `tuhjad_read_audit.py` (kohalik HF cache) | −1 684 (Escher, semper, Pyl, …) |
| Zürich | `audit_zurich.py` (HF, ainult xml-veerg) | **999/1 000 Matcher, ≥2 tühja → välja** (1 jääb) |
| Bullinger | `tuhjad_read_audit.py --kaug` | ehitatud uuesti (`build_bullinger_v3.py`), 709 |
| AAEB, Königsfelden | `--kaug` | AAEB puhas; Königsfelden −3 |
| Hanse XVI / XVII | `tuhjad_read_audit.py` | puhas |
| Svea, Krigshovrätt, Bergskollegium ×2, Göta, Jönköping | `audit_riksarkivet.py` (ainult page_xmls tar) | puhas, kõik read seotud |
| Trolldomskommissionen | `audit_riksarkivet.py` | −19 |
| Senatsprotokolle | ad hoc (ubtue GitHub sparse) → `tuhjad_read_senats.csv` | puhas 229/229 |
| Dresdner 1665 | silmaga + TEI | **vana ehitaja katki** → `build_dresdner_v2.py`, 166 |
| DTA Kosmos / Geusau / Sanders | TEI täistekst (pole TextLine'e) | gap/tabel/U+FFFC lehed välja; struktuurne sõelumine ootel |
| Escher-lisa | `tuhjad_read.csv` | ainult ≤1 tühi |
| vutt_horedad | VUTT „Valmis" (inimese kinnitatud) | — |

Toor-XML (kohalik): `~/_kustutamiseks_20261002/hf_cache/` (xix, hanse),
`data/raw_xml/riksarkivet/` (HF cache), `data/raw_xml/ubtue-gt/`,
`data/raw_xml/hofdiarium1665.xml`, `data/raw_xml/dta_komplett/`,
`data/raw_xml/zh_rrb/` (Zürichi editsioon, vt allpool).

### Dresdner 1665: vana ehitaja oli katki

`build_dresdner_tei_dataset.py` lamendas TEI: tabeli veerud segunesid ühele
reale (lk 00000016), `pb` piiril lekkis järgmise lehe tekst sisse (`pb` on sageli
`persName`-i sees; lk 16 lõppes „Und ſchanckte vor S", mida lehel pole) ja
marginaalid (`note`, 296) visati ära. `build_dresdner_v2.py` kasutab DTA
konverteri `Lehed`-klassi (pb mis tahes sügavusel) ja:
`<ex>` (toimetaja laiend) → „." (lehel on laiendi kohal lühendusmärk: „Se. Churf.
durchl."; nii kirjutasid transkribeerijad ka tabelilahtrites); tabeliga leht välja
(84); ſ jääb (XVII saj allikates on ſ). 250 diaryEntry-lehest **166**.

### DTA käsikirjad: osalise GT asemel täistekst

Matcheri allikas on Deutsches Textarchiv (DTA). DTA TEI kannab sama käsikirja
**täielikku lehe teksti**; treenime lehe tasemel, nii et reajoondust ei ole
vaja. `<pb facs="#fNNNN">` = xix failinime lehenumber (parthey 801 = 801,
hufeland 172, nn_msgermqu 175/341 — kontrollitud silmaga parthey 673).

**Kosmos** (`--grupp kosmos` → `data/dta_kosmos/`, allikas `dta_kosmos_1827`):
loend HU Berlin „Nachschriften der Kosmos-Vorträge". 9 teost, CC BY 4.0:
`parthey_msgermqu1711_1828`, `hufeland_privatbesitz_1829`, `nn_msgermqu2124_1827`,
`nn_msgermqu2345_1827` (olid xix-is Matcheriga) + 5 uut kätt: `libelt_hs6623ii_1828`,
`patzig_msgermfol841842_1828`, `willisen_humboldt_1827`, `nn_oktavgfeo79_1828`,
`nn_n0171w1_1828`. 3 526 lehest **2 463**. `riess_f2e1853_1828` on DTAQ-s
(avalikult ei saa); Lohde ja Stenmark ei ole DTA-s.

**Lisa** (`--grupp lisa` → `data/dta_lisa/`): DTA täiskorpuse
(`dta_komplett_2026-02-10.zip`) käsikirjad — `basisformat_ms` / `handNote`:
- `geusau_reisetagebuchHeinrichxiReuss_1740` → `dta_geusau_1740`, **707** lk:
  Heinrich XI. Reußi reisipäevik, kiire isiklik Kurrent, prantsuse kohanimed
  antiikvas, marginaalid. Silmaga kontrollitud lk 423. Pildi allservas
  digiteerija URN-riba (GT-s pole — mudel õpib eirama).
- `sanders_*` (172 kirja) → `dta_sanders_1860`, **386** lk: Daniel Sandersi kirjad 1859–80.
- Humboldti kirjad Sömmerringile 1791/95: DTA-s pilte pole (404).

Allalaadimine: TEI `https://www.deutschestextarchiv.de/book/download_xml/<id>`
(küpsis `verified=1`) või täiskorpuse zip (`/download`); pildid
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
- **leht välja**, kui seal on `gap` (loetamatu/kadunud), tabel, U+FFFC
  (esitamatu lühendusmärk; enim Patzig) või < 5 rida
Proovida `--naita <id> <nr>`.

### Ehitatud: andmestik v4 (2026-10-04, lõplik)

Iga samm oma kausta (pildid hardlink'itud, sisend puutumata):

| samm | skript | kaust | lehti |
|---|---|---|---|
| v2 miinus ≥ 2 tühja reaga xix-lehed | `build_kurrent_v3.py` | `data/kurrent_v3/` | 18 908 → 17 224 |
| Bullinger: parim XML-versioon, ≤ 1 tühi rida | `build_bullinger_v3.py` | `data/bullinger_v3/` | 707 |
| DTA Kosmos | `build_dta_kosmos.py` | `data/dta_kosmos/` | 2 463 |
| DTA lisa (Geusau, Sanders) | `build_dta_kosmos.py --grupp lisa` | `data/dta_lisa/` | 1 093 |
| Escheri kõik puhtad unikaalsed lehed | `build_escher_lisa.py` | `data/escher_lisa/` | 567 |
| Dresdner uuesti | `build_dresdner_v2.py` | `data/dresdner_v2/` | 166 |
| koondamine + audit-filter | `build_kurrent_v4.py --uuesti` | **`data/kurrent_v4/`** | **18 973** |

Koondamisel: audit −2 363 (Bullinger 1 342, Zürich 999, Trolldom 19,
Königsfelden 3), vana Bullinger −493, vana Dresdner −238, Matcheri DTA-jäägid
−150; lisandub 707 + 2 463 + 1 093 + 567 + 164.

Escher: kasutaja vaatas pildid üle (04.10) — eri käed, VUTT-ile lähedased;
v2 Escheri lagi 1 000 kaotati. Unikaalseid transkribeeritud lehti 3 819, neist
puhtaid (≤ 1 tühi, ≥ 5 rida) 768 — rohkem ei ole (ülejäänud ~9 700 toorkirjet
on transkribeerimata lehed).

Allikad v4-s (lehti): xix_read_1750_99 2 564, **dta_kosmos_1827 2 463**,
aaeb 1 982, xix_read_1800_49 1 793, bergskollegium_rel 1 439, hanse_xvii 1 292,
xix_read_1850_99 1 287, hanse_xvi 1 144, svea 847, trolldom 742, bullinger 709,
**dta_geusau_1740 707**, xix_read_1900 559, **dta_sanders_1860 386**,
krigshovratt 343, senats 229, dresdner 166, dateerimata 101, jonkopings 57,
bergskollegium_adv 53, gota 51, koenigsfelden 31, vutt_horedad 27, zurich 1.

**Holdout 133 → 130**: v3 −5, v4 −18 (Zürich 10, Bullinger 7, Dresdneri tabel 1),
**+20 DTA** (kasutaja 04.10: muidu ei mõõdeta just VUTT-ile lähimaid käsi):
Geusau 10 + Kosmos 10, iga 9 käsikirja vähemalt korra (`lisa_holdout`, seed 3407,
≥ 200 märki nagu `make_holdout.py`). Treeninguks **18 843**.
Vana↔uue mudeli võrdluseks: vanad 73 miinus 10 Zürichi = 63 lehte.

Kontrollitud: 18 973 unikaalset failinime, 0 puuduvat pilti, 0 tühja teksti,
130/130 holdout-rida CSV-s.

### Lühendusmärk → makron (VUTT ADR 0062, #533)

`build_kurrent_v4.py` rakendab kõigile ridadele (sh holdout) `scripts/lyhend_makron.py`
`makroniks`-i: ladina tähe kohal U+0303 tilde ja U+0305 ülakriips → U+0304 makron,
topeltmakron → üks, väljund NFC. Puutumata: kreeka (U+0342), numbrite vinculum,
eraldiseisev „~". Muudetud 4 021 märki (Kosmos 2 338, Sanders 1 075, xix 1800–49 410,
Königsfelden 117, Geusau 40, xix 1850–99 22, Dresdner 19) + 55 rida ainult NFC-ga
(lahutatud „e + U+0304" Königsfeldenis, Hanses, Senatsis). Pärast: 0 tildet, kõik NFC.
Mõõtmises tilde ≡ makron: `eval_kurrent.py` (GT ja väljund läbi `makroniks`-i enne
CER-i) ja `textmetrics.normaliseeri` — vana mudel kirjutab tilde.
Test: `venv/bin/python scripts/lyhend_makron.py --test`. Kaart PEAB kattuma VUTT-i
korpuse teisendusega (#533 samm 2).

### Tootmismudel ja ridade vahelejätmine (PARANDATUD)

Esialgne väide „ridade vahelejätmise tõi alles v2" oli **vale**: 29.08
andmestikus oli 8 000 Zürichi lehte, kõik Matcheriga — viga on mõlemas mudelis.
Eval-väljunditest (GT rida ≥ 8 märki, mille parim vaste väljundis < 0,6): vanad
73 holdout-lehte — `kurrent-20260829` 29 puuduvat rida, `kurrent-20261002` 31 =
viik; kõigil 128-l 177 → 48. See võrdlus ise oli osaliselt vigase GT peal
(10 Zürichi lehte vanadest 73-st) → tagasivahetust ei ole vaja, aga mõlemad
mudelid jätavad tõenäoliselt ridu vahele just tabelites ja marginaalides.

### Zürich: täistekst on olemas (järgmine iteratsioon)

Zenodo 10517999 „TEI-XML Zürcher Regierungsratsbeschlüsse 1803–1887"
(CC BY-SA 4.0, 166 755 otsust, `data/raw_xml/zh_rrb/`): Wordi-transkriptid,
täielik tekst, tabelid rida-realt, `<pb n="308"/>` täpselt õiges kohas — lk 308
kõik 26 puuduvat rida on olemas. AGA: jooksvas tekstis reavahetusi pole,
marginaalpealkiri on päises `<title>` (normaliseeritud), kuupäevapäis ainult
`1807-05-28`. Taastamine: Matcheri õiged täidetud read ankruteks (proosa;
tabeliread on Matcheris valed → geomeetria järgi), ankrute vahe editsioonist
tühjadele ridadele; lehed, mida ei õnnestu üheselt taastada, välja. Kasutaja:
käsi on relevantne, ebatraditsiooniline lehekuju aitab. ~pool päeva + kontroll.
`dh-unibe/image-text_zh-regierungsratsprotokolle` (HF) on SAMA Matcheri materjal.

### Vaadatud 04.10, mitte kasutusel

- Zenodo 17252677 pages/: uusi lehti ~27 (Auerbach, Erbkam, Baieri Schriftkunde) — liiga vähe.
- `DenisaBumba/htr_leibniz_dataset_v1`: Leibniz, XVII saj ladina/prantsuse
  õpetlaskäsi — VUTT-i jaoks väga asjakohane, aga reataseme ja osaliselt
  automaatjoondus; lehe tasemel treeningusse ei sobi otse.
- dh-unibe 1848+ kogud (Bundesratsprotokolle jms): kantseleikäsi, sama ühekülgsus mis Zürich.

### GT-kontroll ja väljavõtt (05.–06.10, tehtud)

Tootmismudel (`kurrent-20261002-Q8_0`) üle kogu v4 treeningkomplekti (18 843 lk,
`scripts/gt_kontroll.py`, holdout väljas). CER ≥ 10 %: 1 403 kandidaati.

CER on ridade järjekorrale tundlik — mitmeveeruline leht, marginaalia või tabel
annab kõrge CER-i ka siis, kui sisu klapib. `scripts/gt_sonakate.py` võrdleb
sõnahulka (recall/precision) ja klassifitseerib: sisu klapib 931, piiripealne 307,
loop 37 (mudeli rike, GT korras) → jäävad; **välja 127**: erinev 82 (enamasti
parandamata HTR-GT, kasutaja kinnitas pildilt aaeb 16388), GT puudulik 37,
osaline GT 8. Väljavõtt: `scripts/kurrent_gt_valja.py --kirjuta` →
`metadata.csv` 18 973 → 18 846, treeningule **18 716** (varukoopia `metadata.csv.bak`).

Leiud allikate kaupa:
- **aaeb_xiv_xvii** on ~99 % prantsuse keeles; jääb sisse (ladina kiri esineb ka
  Kurrendi tekstides; 20261002 mudel loeb seda juba hästi, med 1,6 %). 59 lk välja.
- **dresdner_1665** med 16,4 % on MUDELI viga: v2 vana TEI-ehitaja laiendused
  („Churf ürstliche d urc hl aucht", tühik enne koma) on mudelis sees. GT õige;
  `gt_sonakate` erand „osaline GT"-st. Uus treening peaks harjumuse kaotama —
  vaata evalis.
- **bullinger_autoren v3**: aadresslehtede kõrge CER = arhiivimärgid („S.", „1.")
  GT-s, mida mudel ei loe. Toores HTR ainult 7 lehel (välja).
- **dta_kosmos_1827**: loengukonspektide ridadevahelised lisandused → järjekord.
- **vutt_horedad** kaitstud: tahtlikult peaaegu tühjad lehed (nt VUTT jbc88t lk 4
  „2v") hallutsineerimise vastu. Mudel jätab harilikuga foliandi lugemata.
- DTA holdout 20 lk (mudel vs GT, pikkused klapivad): CER 0,6–4,5 %, v.a
  libelt_0291 15,5 % — diplomaatiline GT lühenditega („ud", „Thren"), mudel
  laiendab. Kõik korras.

Toor-XML (`~/_kustutamiseks_20261002/hf_cache/`) **ära kustuta enne uut
treeningut**. Vana v2: `~/_kustutamiseks_20261005/data_kurrent_v2`.

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
