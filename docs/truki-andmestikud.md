# Trüki-OCR treeninguandmestikud

**Viimati uuendatud:** 2026-07-08

Trüki-OCR treeningahel koosneb kahest etapist (vt ka `MEMORY.md` → "Mudel ja treeningahel"):

1. **Baastreening** (puhas transkriptsioon, ilma markupita) → `models/qwen3.5-ocr-lora-backup-20260527`
2. **Markup-treening** (VUTT XML: `<m>`, `<i>`, `<cs>`) sellest baasist edasi → `models/qwen3.5-ocr-markup-20260614` (praegu aktiivne)

---

## 1. Baastreening — `data/lehekyljed/metadata.csv`

**1 500 lehekülge**, 2 veergu (`failinimi`, `transkriptsioon`), **allika-veergu pole**.
Fail on loodud **11.11.2025** ja pole sellest ajast muutunud — nii 27.05 kui 14.06.2026 treening kasutasid täpselt sedasama, praegu nähtavat andmestikku. Ehitusskripti ei ole (erinevalt Kurrent-andmestikust) — kaust on ilmselt käsitsi/eri allalaadimiste tulemusel kokku pandud.

Failinimede järgi tuvastatav jaotus:

| Allikas | Lk | Keel/kiri |
|---|---|---|
| **Gezelius'e Kreeka-Ladina leksikon** | 720 (48%) | kreeka + ladina |
| **Gallicorpora** (`bpt*`, `data_*`) — HTR-imprime-17e-siecle | 242 | prantsuse |
| **Nummerdatud skaneeringud** (`00010.jpg` jms, allikas tuvastamata) | 140 | — |
| **Tartu Academia Gustaviana disputatsioonid** (`r_acad_dorp_`, `r_est_a_`) | 81 | ladina |
| **Arrhenius-Ekman** (disputatsioon) | 44 | ladina |
| **1679 Gartman** (disputatsioon) | 44 | ladina |
| **Liungh-Celenius** (disputatsioon) | 36 | ladina |
| **Schönlandt, Templa...** | 25 | ladina |
| **Figrelius-Svebilius** | 24 | ladina |
| **Image_*** (tuvastamata disputatsioon) | 17 | ladina |
| **Cederschiöld-Hyltén** | 16 | ladina |
| **Drucke_qun** | 15 | ladina |
| **De origine Livonorum** | 14 | ladina |
| **IMG_2019* fotod** (käsitsi pildistatud lehed) | 13 | ladina |
| **PDFsam_ocm05*** (PDF-ist lõigatud skaneeringud) | 10 | — |
| **Zoologia Danica** | 7 | ladina |
| Muu/üksikud | 52 | — |

15 rida on tühja transkriptsiooniga (treeningskript filtreerib need automaatselt).

**Märkus:** `data/raw/` sisaldab osaliselt samu Tartu disputatsioone + Jõgever_Ilias fotosid, aga ka Rootsi Riksarkivet tar.gz-e — need viimased on tegelikult **Kurrent/käsikirja** andmestiku toorfailid (vt `kurrent-andmestikud.md`), mitte trüki-OCR jaoks. `data/processed/` (7868 rida, `scripts/prepare_raw.py` väljund) on omakorda kitsas 10-dokumendiline alamhulk, mis **ei ole** 1500-lk baasandmestiku allikas.

### Allikate ja teksti autorlus/rahastus

- **Gezelius'e Kreeka-Ladina leksikon** — valminud ETAg uurimisprojekti **PUT 132** raames ("Humanist Greek in Early Modern Estonia and Livonia: A Cultural Bridge to the European Present and Past", 2013–2016, PI J. Päll) ning projekti **Helleno-Nordica: Humanist Greek in the Swedish Empire** raames (PI Janika Päll, Lundi Ülikooli projekti "Helleno-Nordica. The Humanist Greek Heritage of the Swedish Empire" alamprojekt, PI Johanna Akujärvi, 2016–2022), samuti Tartu Ülikooli baasfinantseeritud grantide toel.
- **Gallicorpora** (`bpt*`/`data_*`, prantsuse tragikomöödiad) — [github.com/Gallicorpora/HTR-imprime-17e-siecle](https://github.com/Gallicorpora/HTR-imprime-17e-siecle)
- **Kõik ülejäänud materjal** (Tartu Academia Gustaviana disputatsioonid, orationes jm) on transkribeeritud käsitsi ja Transkribus/eScriptorium keskkondades. Transkriptsiooni autorid ja retsensendid (tähestiku järjekorras): Ove Averin, Meelis Friedenthal, Jaana Jurtšenkova, Kristiina Kase, Vallo Kask, Kristin Klaus, Hant Mikit Kolk, Pärtel Piirimäe, Agne Pilvisto, Anni Polding, Janika Päll, Kaarina Rein, Rahel Toomik.

---

## 2. Markup-treening — `data/vutt/metadata.csv`

**749 lehekülge**, 65 unikaalset teost, valdavalt **Tartu Academia Gustaviana/Gustavo-Carolina** ladinakeelsed disputatsioonid/orationes (17.–18. saj), nüüd VUTT XML-märgendusega (`<m>` marginaalid, `<i>` kaldkiri, `<cs>` koodivahetus). Sama autorluse/transkriptsiooni taust kui sammu 1 "ülejäänud materjalil" (vt eespool) — käsitsi + Transkribus/eScriptorium, sama nimekiri transkriptsiooni autoreid/retsensente.

Suurimad teosed:

| Teos | Lk |
|---|---|
| 1690 Liber philosophus | 50 |
| 1696-30 Exercitatio politica de majestate... | 47 |
| 1647-1 Oratio panegyrica... | 43 |
| 1706 Wilde, Templa... | 25 |
| 1633-15 Encomiasticon... | 24 |
| 1707-17 Dissertatio philosophica de eo quod est physicum... | 24 |
| 1693-7 De origine Livonorum dissertatio... | 23 |
| 1696-9 Dissertatio philosophica de trinitate Platonis... | 22 |
| ülejäänud 57 teost | 3–21 |

**Andmevoog (alates 04.08.2026):** öine `vutt_backup.py` cron → rsync-snapshot `~/vutt-backups/latest/data` (1317 teost) → `scripts/build_vutt_dataset.py` filtreerib ainult **"Valmis"** staatusega lehed, eemaldab tühjad `<m>` sildid → `data/vutt/metadata.csv` + `data/vutt/SOURCE.txt`.

**Vana andmevoog (kuni 04.08.2026):** `scripts/vutt_sync.py` → `data/vutt-raw/`. Aegunud ja kustutatud. See tõmme jooksis ilma `--delete`-ita, seega serverist kustutatud lehed jäid kohalikku koopiasse alles — 04.08 võrdlusel oli `vutt-raw`-s ~15 lehte, mida serveris enam ei ole. Snapshot on autoriteetne.

**Ajaline järjestus (kontrollitud):** `data/vutt/metadata.csv` mtime = 14.06.2026 22:58; treeningu checkpoint (`models/qwen3.5-ocr-markup-20260614/adapter_model.safetensors`) mtime = 23:46 samal õhtul. Andmestik on treeninguga konsistentne — praegune 749-lk fail on täpselt see, mis treeningusse läks (erinevalt Kurrent-andmestikust, kus see nii ei olnud, vt `kurrent-andmestikud.md`).

**NB:** 04.08.2026 ehitati andmestik uuesti, seekord backup-snapshot'ist `20260804T143956Z`: **1113 Valmis lehte** (749 → 1066 → 1113). See ei mõjuta juba tehtud 14.06 treeningut, vaid on materjal järgmisele voorule. Iga ehitus kirjutab nüüd `data/vutt/SOURCE.txt`, kust näeb kasutatud snapshot'i.

**Lahknevus koodis:** `scripts/train_markup.py` docstring (read 12–13) väidab, et andmed tulevad kombineeritult `data/processed/` (136 lk, "vanem käsitsi märgendatud materjal") + `data/vutt/`, aga tegelik `DATA_SOURCES` muutuja (rida 55) sisaldab **ainult** `data/vutt/metadata.csv`. `data/processed/` (loodud märts 2026, sisaldab `*kursiiv*`-markupit) **ei ole** tegelikult 14.06 treeningus kasutatud, ehkki docstring seda väidab — docstring tuleks parandada või data/processed uuesti lisada.

## Lühendusmärk → makron (VUTT ADR 0062, #533; 05.10.2026)

`scripts/lyhend_makron_trukk.py` (kuivkäivitus vaikimisi, `--kirjuta`, `--test`)
teisendas `data/lehekyljed/metadata.csv`, `metadata_markup.csv` ja `data/vutt/metadata.csv`:

| | lehekyljed (kumbki CSV) | vutt |
|---|---|---|
| tilde/ülakriips → makron (märke) | 1 018 | 1 011 |
| U+E8BF → `q;` (MUFI que-ligatuur; `q;` on andmestiku valitsev kuju, ~6 000×) | 53 | 3 |
| U+F1A7 → `I` (kursiivne suur I, pildilt kontrollitud) | 1 | 0 |
| kombineeriv märk rea alguses → maha | 2 | 1 |
| kreeka tilde → U+0342 | 1 | 0 |
| muutunud lehti (sh NFC: 666 lehte lagundatud aktsentidega) | 740 | 387 |

Keelevalvurit ei olnud vaja: `data/vutt` 962/1120 lehe teose `languages` on valvurita,
ülejäänud 158 on kärbitud kaustanimega 1626–1802 ladina/saksa teosed; `data/lehekyljed` `õ`-d
on kõik lühendid. `build_vutt_dataset.py` rakendab sama `makroniks`-i igal ehitusel, v.a teosed,
mille `languages` ∋ est/spa/por (need loetletakse). Mudeli väljundeid (`data/vutt/eval`,
`reocr`) ei muudetud — `eval_print.py` ja `textmetrics.normaliseeri` võrdsustavad tilde ja makroni.

**Jäi alles (otsustamata):** `Arrhenius-Ekman_De_iure_Dei_in_homines_1772.pdf_page_25/41`
kreeka read on GT-s rämps (`didῦoV`, `ꝙλλὰ`) — kandidaadid välja jätmiseks.
`ꝗ` (5×, prantsuse „qui") on MUFI-standardne märk ja jäi.

---

## Seotud, aga (veel) treeningus kasutamata

| Kaust | Kirjeldus | Seis |
|---|---|---|
| `data/freilingshausen-2col/` | Freylinghausen Gesang-Buch (1741), BSB IIIF, 1180 lk, saksa, 2-veeruline küljendus | Loodud 01.07.2026 — ette valmistatud VUTT-üleslaadimiseks (`vutt-upload/`: few-shot-OCR eeltranskriptsioon + jpg, inimkorrektuuri jaoks). Ei ole veel üheski treeningus. |
| `data/test/` | Käsitsi kureeritud eval-komplekt: üksikud lehed + `ground-truth/` + mudeliversioonide (20260526/531/614 jt) väljundid | Hindamiseks, mitte treeninguks. |
