# Kurrent-mudeli strateegia ja praegune seis

**Viimati uuendatud:** 2026-06-03

---

## Eesmärk

Üks käsikirjaline mudel, mis katab meie materjali vahemikus XVII–XIX sajand:
- Saksa Kurrent (XVI–XIX saj) – Hansaliit, Herrnhut, Tartu ülikooli kirjavahetus
- Rootsi kantseleikirjutus (XVII–XVIII saj) – Riksarkivet allikad
- Ladina humanistlik kurssiiv (XVII–XVIII saj)

Trükimudel (qwen3.5-ocr-lora-backup-20260527) jääb eraldi.

---

## Praeguse mudeli diagnoos

**Mudel:** qwen3.5-ocr-kurrent-20260602 (treenitud 16 579 lk peal)

Testitud kolme dokumendiga:

| Dokument | Periood | Keel | Tulemus |
|----------|---------|------|---------|
| 1689 Tartu ülikooli Rootsi kuninglik hart | XVII saj | Ladina | Suurepärane |
| 1690 Tartu ülikooli senatiprotokoll | XVII saj | Rootsi | Väga hea |
| 1759 Herrnhuti Diasporae diarium (Dorpat) | XVIII saj | Saksa | Rahuldav – süstemaatilised vead |

**Järeldus:** Kirjastiil ei ole probleem. Mudel loeb XVIII saj Kurrenti tähekujusid õigesti. Vead on sõnavara- ja lühendispetsiifilised:
- Herrnhuti lühendisüsteem (`Hld` = Heiland, `Lg.` = Liebling, `L.` = Lieber/Liebste, `Br./Sr.` = Bruder/Schwester) – mudel ei tunne
- Eesti kohanimed saksa kujul – mudel arvab vale sõna
- Lühendite laiendamine: mudel hallutsineerib plausiiblise saksa sõna (nt `flt. rel.` → `silvestri`)

---

## Miks XVIII saj saksa andmestikku pole

Otsisime läbi:
- **HuggingFace** – kõik relevantsed saksa Kurrenti andmestikud on kasutusel (vt `kurrent-andmestikud.md`)
- **HTR-United** (GitHub) – 51 andmestikku, saksa materjali ainult 8; XVIII–XIX saj saksa Kurrenti pole ühtegi
- **Transkribus** – 12 avalikku andmestikku, ükski ei kata meie teemat

Leid HTR-Unitedist: kaks lisandatavat Dresdner Hofdiarium datasetti (1653–56 ja 1665), mis pole veel meil kasutusel. Mõlemad XVII saj, sarnane stiil olemasoleva 1673 andmestikuga – väikesed (~20 lk kumbki) aga legitiimne XVIII saj saksa eelne materjal.

**Põhijäreldus:** XVIII saj saksa Kurrenti avalikke treeningandmeid ei eksisteeri. See on teadaolev lünk, mitte otsinguprobleem.

---

## Praeguse andmestiku probleem

kurrent_xix domineerib 48%-ga (8 000 / 16 579 lk). Mudel "mõtleb" XIX sajandil.

Tegelik katvus pärast tasakaalustamist (eesmärk ~1 000–1 500 lk/allikas):

```
XVI saj    hanse_kurrent_xvi, bullinger_autoren, aaeb_xiv_xvii (lõpuosa)
XVII saj   aaeb_xiv_xvii (algus), dresdner (3 datasetti), Riksarkivet allikad
XVIII saj  Riksarkivet allikad (rootsi), Herrnhuti materjal (tulemas)
XIX saj    kurrent_xix (tasakaalustatud), koenigsfelden
```

---

## Tegevuskava

### 1. Kohene – andmestiku tasakaalustamine
- `scripts/filter_dataset.py --max-per-source 1500 --out data/kurrent/metadata_balanced.csv`
- Kurrent_xix langeb 8 000 → 1 500, teised allikad jäävad puutumata

### 2. Lähiajal – Dresdner lisandused
- Dresdner Hofdiarium 1665: zenodo.14356190
- Dresdner Hofdiarium 1653–56: zenodo.15303398
- Lisada `scripts/build_dresdner_dataset.py` kaudu (ALTO XML formaat, sama loogika)

### 3. Herrnhuti materjal (käsil)
- Esimene pass: Gemini Flash API + käsitsi parandus
- Lühendite tabel kaasa prompts'is (Hld, Lg., L., Br., Sr. jt)
- Eesmärk: ~50–100 lehekülge korrektseid transkriptsioone
- Lisada treeningandmetesse → uus treeningutsükkel

### 4. Uus treening
- Lähtepunkt: qwen3.5-ocr-kurrent-20260602 (mitte baas-Qwen)
- Andmestik: tasakaalustatud metadata_balanced.csv + Herrnhuti lehed
- Parameetrid: sama mis eelmises treeningus (r=64, 2 epohhiga)

---

## Mis ei vaja muutmist

- Ladina ladina-humanistlik kurssiiv: mudel töötab hästi juba praegu
- Rootsi XVII–XVIII saj: Riksarkivet allikad annavad piisava katvuse
- Trükimudel: eraldi, ei puuduta käsikirjalist mudelit
