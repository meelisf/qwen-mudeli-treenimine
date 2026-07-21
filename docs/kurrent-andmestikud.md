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

## Vaadatud aga mitte kasutusel

| Andmestik | Põhjus |
|-----------|--------|
| [Riksarkivet/frihetstidens_utskottshandlingar_seg](https://huggingface.co/datasets/Riksarkivet/frihetstidens_utskottshandlingar_seg) | Tühjad transkriptsioonid – segmenteeritud aga transkribeerimata |
| [Zenodo 10.5281/zenodo.17252677](https://zenodo.org/records/17252677) – German Kurrent HTR 9317 rida | Reataseme andmestik (lõigatud read), mitte täisleheküljed; kattub kurrent_xix-ga |
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
