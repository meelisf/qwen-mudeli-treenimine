# Märgenduskatvuse katse — tulemused (28.–29.08.2026)

**Mudel:** `models/qwen3.5-ocr-print-base-r64-mi-vl-20260828`
**Kontrollrühm:** `models/qwen3.5-ocr-print-base-r64-20260827` (praegu tootmises)

## 1. Mida katsetati

Kaks muutujat korraga, sest need on sisuliselt sama asi — **parem märgenduse
katvus treeningandmetes**:

1. `--keep-m-italics` — `<i>` jääb `<m>` sisse alles (varem `strip_italics_in_marginalia()`
   võttis selle maha, mistõttu `<m>` ja `<i>` võistlesid; vt `marginaal-kursiiv-kaks-telge`).
2. `--valitud-lehekyljed` — 1. etapi märgendamata lehed jäävad välja. Sisse jäid
   ainult need, mille tag-null on **aus**: Gezelius Lexicon 21–440 (sünteetiliselt
   märgendatud) ja Comenius Ianua (pole marginaale ega kursiivi).

**Küsimus, millele vastust otsiti:** kas märgendamata lehtede treeningusse
võtmine õpetas mudelit `<m>` ära jätma? Kasutaja tähelepanek: „vahel märgendab
päris hästi, ja vahel ei märgenda, aga transkribeerib neid küll."

## 2. Treening

| | |
|---|---|
| Käsk | `train_markup.py --base=unsloth/Qwen3.5-9B --lora-rank=64 --keep-m-italics --valitud-lehekyljed` |
| Näiteid | **1 793** (kontrollrühmal 2 578) |
| Välja jäetud 1. etapist | 807 lehte (640 „muu", 132 märgendamata Becker, 27 ettevalmistamata Lexicon) |
| Sammud / epohhid | 450 / 2 |
| Aeg | 12 900 s ≈ 3,58 h (19:01 valmis) |
| train_loss | **0,061** (lõppsammudel 0,017–0,024) |

Fraktuuri osakaal langes: 138 → 72 ⸗-lehte, `⸗` 1 658 → 399.

## 3. VUTT holdout (20 lehte) — `eval_print.py`

Kontrollrühm on **sama GT-ga ümber skooritud** (`--keep-m-italics --resume
--name 20260827-migt`), muidu poleks CER võrreldav.

| | vana (20260827) | **uus (mi-vl)** |
|---|---|---|
| CER kokku | 3,4 % | **1,4 %** |
| CER mediaan | 2,0 % | **1,2 %** |
| CER märgenditeta (`cer_plain`) | 0,9 % | 0,8 % |
| `<m>` väljundis (GT 185) | 170 | **181** |
| `<m>` täistabamusi | 150 | **156** |
| `<m>` F1 | 0,79 | **0,81** |
| `<m>` sisu-CER | 13,1 % | **9,5 %** |
| `<i>` väljundis (GT 270) | 103 | **242** |
| `<cs>` väljundis (GT 16) | **16** | 11 |

Kihtide kaupa (uus mudel):

| kiht | lk | CER | CERtxt | `<m>`gt | `<m>`out | F1 | m_CER |
|---|---|---|---|---|---|---|---|
| marginaalirohke | 10 | 1,8 % | 0,9 % | 176 | 174 | 0,88 | 4,9 % |
| marginaaliga | 5 | 1,5 % | 0,9 % | 9 | 7 | 0,50 | 28,0 % |
| ilma | 5 | 0,4 % | 0,4 % | 0 | 0 | 1,00 | 0,0 % |

**Kuidas seda lugeda.** Suur CER-i võit (3,4 → 1,4 %) tuleb valdavalt `<i>`-st
`<m>` sees: `<i>` 103 → 242. Vana mudelit karistatakse siin muutuja eest, mida
talle kunagi ei õpetatud — see osa võidust on osalt artefakt. **Aus võrdlus on
`cer_plain` (0,9 vs 0,8 %) — transkriptsioonikvaliteet jäi samaks** — ja
`<m>` mõõdikud, mis paranesid päriselt.

**Üks regressioon:** `<cs>` 16 → 11. Väike arv, aga jälgida.

## 4. Menii sond (13 lehte) — otsustav test

Menii on treeningust praktiliselt väljas (1 leht 1 793-st, sondi lehed 0020–0048
puutumata). Siin nägi vana mudel veeru ära, aga jättis `<m>` panemata.

**`<m>` kokku 13 lehel: 70 → 207 (≈ 3×).**

| leht | vana `<m>` | uus `<m>` | vana `<i>` | uus `<i>` |
|---|---|---|---|---|
| 0020 | 12 | 10 | 14 | 26 |
| 0024 | 31 | 22 | 0 | 19 |
| 0025 | 0 | 0 | 38 | 80 |
| 0026 | 0 | **28** | 0 | 33 |
| 0027 | 0 | 0 | 0 | 38 |
| 0028 | 0 | **33** | 60 | 53 |
| 0029 | 0 | 0 | 0 | 26 |
| 0030 | 0 | **22** | 34 | 35 |
| 0031 | 0 | **19** | 19 | 19 |
| 0037 | 0 | 0 | 0 | 16 |
| 0038 | 27 | 27 | 9 | 35 |
| 0043 | 0 | **26** | 2 | 37 |
| 0048 | 0 | **20** | 0 | 0 |

- **6 lehte taastus täielikult** (0026, 0028, 0030, 0031, 0043, 0048): 0 → 19…33 `<m>`.
- **4 lehte jäid nulli** (0025, 0027, 0029, 0037) — aga **veaviis on muutunud**.
  Uus mudel **loeb marginaaliveeru täielikult välja** ja paneb iga rea eraldi
  reale `<i>`-sse; puudu on ainult väline `<m>`:

  | leht | ainult-`<i>` ridu, vana | ainult-`<i>` ridu, uus |
  |---|---|---|
  | 0025 | 12 | 28 |
  | 0027 | 0 | 29 |
  | 0029 | 0 | 19 |
  | 0037 | 0 | 10 |

  Näide lehelt 0029: `<i>Cymbri Sar-</i>` / `<i>des Lydiæ</i>` / `<i>Metropolin</i>` …
  — see on marginaaliveerg rida-realt, ainult vale tagiga.

- **Kaks lehte läksid alla:** 0020 (12 → 10) ja 0024 (31 → 22). 0024-l kaob
  pikast nimeloendist 9 kirjet (Careotæ, Cimbri, Eſty, Galindæ, Gothi,
  Gythones, Igilliones, Lemovij, Livones) — päris recall-kaotus.

**Järeldus:** jah, märgendamata lehtede sisse võtmine õpetas mudelit `<m>`-i
ära jätma. Selle eemaldamine taastas 6 lehte 10-st ja kolmekordistas `<m>`
arvu. Jääk-viga ei ole enam „ei näe veergu", vaid **`<i>` võidab `<m>` üle** —
täpselt see telg, mille kohta hüpotees oli.

## 5. Fraktuur — `becker_probe.py` (28 lehte)

Kontrollküsimus: kas 132 märgendamata Beckeri lehe väljajätmine lõhkus fraktuuri?

**Lehed 1–8 (VUTT GT, mõlema treeningus):**

| mudel | cer_plain | `<cs>` GT | `<cs>` väljund |
|---|---|---|---|
| 20260827 | 0,6 % | 37 | 0 |
| mi-vl-20260828 | 1,3 % | 37 | 0 |

**Lehed 9–140 (20 tk, GT-d ei ole). Vana mudel nägi neid treeningus, uus mitte:**

| mudel | märke/lk | `<cs>` | `⸗` | `<m>` | loope |
|---|---|---|---|---|---|
| 20260827 | 1 669 | 0 | 177 | 0 | 0 |
| mi-vl-20260828 | 1 663 | 1 | 174 | 0 | 0 |

Mudelite omavaheline lahknevus (CER, tagideta): keskm **1,3 %**, mediaan 0,9 %.
Suurim: 00037 (2,9 %), 00030 (2,9 %), 00057 (2,6 %).

**Järeldus: fraktuur ei kannatanud.** 132 lehe väljajätmine maksis lehtedel
9–140 sisuliselt mitte midagi (märke/lk 1 669 → 1 663, `⸗` 177 → 174, loope 0),
kuigi vana mudel oli need lehed pähe õppinud ja uus polnud neid näinud.
Lehtede 1–8 cer_plain 0,6 → 1,3 % on samuti mälupõhine ja seetõttu ebaaus
võrdlus — aga isegi see on väike.

Kaks kohta silmaga üle vaadata: **00085**, kus `⸗` läks 7 → 0, ja **00064**,
kus tekkis üksik `<cs>`.

## 6. Kokkuvõte

| telg | tulemus |
|---|---|
| `<m>` katvus Menii peal | **70 → 207**, 6 lehte 10-st taastus |
| `<m>` holdoutil | 170 → 181 / 185, F1 0,79 → 0,81, sisu-CER 13,1 → 9,5 % |
| transkriptsioon (cer_plain) | 0,9 → 0,8 % — muutumatu |
| fraktuur | muutumatu (lahknevus 1,3 %) |
| `<cs>` | 16 → 11 holdoutil — väike regressioon |
| `<i>` | 103 → 242 / 270 — ootuspärane, muutuja ise |

Katse **õnnestus**: hüpotees leidis kinnitust ja mudel on `<m>` telje peal
selgelt parem, ilma et transkriptsioon või fraktuur oleks kannatanud.

## 7. Lahtised otsad

- ~~**Aktiveerimine** on tegemata~~ → **GGUF ja pariteet tehtud 29.08 öösel, §9.**
  Unit-fail ja `MODEL_CONFIGS` osutavad uuele mudelile; jäänud on ainult
  sudo-ga teenuste käivitamine.
- **Teenused seisavad** (`ocr-service`, `llama-server-print`, `llama-server-hand`) —
  vajavad sudo-ga käivitamist. Käsuplokk: `docs/SEIS.md` §1.
- ~~**`<cs>` regressioon** (16 → 11)~~ → §9.3: 147 lehe peal 166 → 160, ehk
  holdouti langus on pigem 20 lehe müra. Jälgi, aga eraldi katset ei nõua.
- **Menii 0025/0027/0029/0037**: `<i>` võidab `<m>` üle. Järgmise katse kandidaat.
- **Menii 0024** kaotas 9 marginaalikirjet pikast nimeloendist.
- **Fraktuuril pole päris GT-d** lehtedele 9–140. Kui tahta päris mõõtu, märgi
  VUTT-is paar neist valmis JA hoia treeningust väljas (`holdout.txt`).
- **Becker 00085** (`⸗` 7 → 0) ja **00064** (`<cs>` +1) — silmaga üle vaadata.

## 8. Kus andmed on

```
data/vutt/eval/qwen3.5-ocr-print-base-r64-mi-vl-20260828/   # holdout, uus
data/vutt/eval/20260827-migt/                               # holdout, vana, sama GT-ga
data/vutt/reocr/menii-probe-qwen3.5-ocr-print-base-r64-mi-vl-20260828/
data/vutt/reocr/menii-probe-qwen3.5-ocr-print-base-r64-20260827/
data/vutt/reocr/becker-probe-*/
data/vutt/eval/print-base-r64-mi-vl-Q8_0/          # holdout, uus GGUF (§9.1)
data/vutt/eval/print-base-r64-Q8_0/                # holdout, vana GGUF, sama GT-ga
data/vutt/reocr/print-base-r64-mi-vl-Q8_0/         # 153 Toores lehte, uus GGUF (§9.2-9.3)
data/vutt/reocr/print-base-r64-Q8_0/               # 147 Toores lehte, vana GGUF
```

Kordamine:

```bash
venv/bin/python scripts/eval_print.py --keep-m-italics models/qwen3.5-ocr-print-base-r64-mi-vl-20260828
venv/bin/python scripts/eval_print.py --keep-m-italics --resume --name 20260827-migt \
    models/qwen3.5-ocr-print-base-r64-20260827
venv/bin/python scripts/menii_probe.py models/qwen3.5-ocr-print-base-r64-mi-vl-20260828
venv/bin/python scripts/becker_probe.py
```

---

## 9. GGUF-i konversioon ja pariteedikontroll (29.08, öö)

`scripts/merge_lora.py` → `convert_hf_to_gguf.py --no-nextn` → `--mmproj`
→ `llama-quantize Q8_0`. Kokku **2,5 min** (merge 03:22 → valmis 03:24).

```
models/gguf/print-base-r64-mi-vl-20260828-Q8_0.gguf         9 527 501 440 B
models/gguf/mmproj-print-base-r64-mi-vl-20260828-F16.gguf     918 165 472 B
models/gguf/print-base-r64-mi-vl-20260828-BF16.gguf        17 920 696 960 B (vahefail)
```

Server käivitati käsitsi `--image-max-tokens 5000 -ngl 99 -c 65536 -np 4 -cb
-fa on` peal, GPU 12,2 GB. **Kärpekontroll:** pildiga päringu
`prompt eval ... / 5783 tokens` ehk ~4 950 visuaaltokenit — täpselt see, mida
§2.1 nõuab (alla ~4 100 tähendaks, et lipp on puudu).

### 9.1 Holdout (20 lehte), kõik neli kombinatsiooni sama GT-ga

Kõik neli rida on skooritud `--keep-m-italics` GT vastu, muidu poleks CER
võrreldav (vana GGUF-i rida on `--resume`-ga ümber skooritud).

| | CER | `cer_plain` | `<m>` /185 | `<m>` sisu-CER | `<cs>` /16 | s/lk |
|---|---|---|---|---|---|---|
| vana, transformers (`20260827-migt`) | 3,4 % | 0,86 % | 170 | 13,1 % | 16 | 30,8 |
| vana, GGUF Q8_0 | 3,2 % | 0,82 % | 169 | 13,1 % | 17 | 6,2 |
| **uus mi-vl, transformers** | 1,4 % | 0,78 % | 181 | 9,5 % | 11 | 33,1 |
| **uus mi-vl, GGUF Q8_0** | **1,5 %** | **0,80 %** | **182** | **9,8 %** | 11 | **6,5** |

**Pariteet on olemas.** Uue mudeli GGUF ja transformers lahknevad `<m>`-is ühe
tagi võrra (182 vs 181) ja CER-is 0,1 pp — sama suurusjärk mis 27.08 mõõdetud
vana mudeli pariteet. Kiirus 5x.

### 9.2 Menii sond GGUF-i ahelas (13 lehte)

Sama 13 lehte, aga **tootmisahelas** (`fit_to_grid` + PNG + llama-server),
mitte `menii_probe.py` toorpildi-ahelas. Read on `reocr_vutt.py` väljundist.

| ahel | `<m>` kokku | lehe kaupa |
|---|---|---|
| vana GGUF | 98 | `0 0 0 0 0 0 0 0 0 0 39 27 32` |
| **uus GGUF** | **205** | `0 0 22 11 20 26 20 19 0 0 37 27 23` |
| (võrdluseks: uus transformers) | 207 | `0 0 22 10 20 26 28 19 0 0 33 27 22` |

Katvusvõit **kandub GGUF-i üle täies mahus** (98 → 205, transformersil 70 →
207). Samad neli lehte (0025/0027/0029/0037 = positsioonid 9, 1, 2, 10) jäävad
nulli mõlemal mootoril — §1b vealiik on mudeli oma, mitte ahela oma.

### 9.3 Lai A/B: 147 ühist „Toores" lehte, vana GGUF vs uus GGUF

`reocr_vutt.py` andis seekord 153 lehte (nimekiri on päring ja kasvab, §4);
võrreldud on 147 ühist.

| | `<m>` | `<i>` | `<cs>` | märke | loope |
|---|---|---|---|---|---|
| vana GGUF | 590 | 1 394 | 166 | 281 670 | 3 |
| uus GGUF | **635** | 3 160 | 160 | 286 168 | 3 |

`<i>` kahekordistumine on katse muutuja ise (`<m>` sees olev kursiiv jäeti
alles), mitte regressioon. `<cs>` 166 → 160 — **holdouti 16 → 11 ei paista
korpuse peal**, ehk §1c on pigem 20 lehe müra kui päris regressioon.

Loope 3 mõlemal, aga eri lehtedel: vana `1638_39_0133`, uus `1638_39_0135`
(mõlemal ühised `…tvh4im-253` ja `1638_39_0044`). Loopiv leht liigub, arv ei
kasva.

### 9.4 Otsus

Konversioon ja pariteet on tehtud, mudel läheb tootmisse. Aktiveerimise
käsuplokk on `SPIKKER.md`-s ja `docs/SEIS.md` §1-s — **nõuab sudo-t**, seega
seda sammu skript ise teha ei saanud.
