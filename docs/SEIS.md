# Seis: mida me teame

**Elav dokument.** Uuenda seda, ära tee uut kuupäevaga faili — just
kuupäevaliste paralleeldokumentide kuhjumine tekitas 28.08 hommikul segaduse,
kus ma kordasin ühest failist järeldust, mille teine fail oli juba ümber
lükanud.

Tööjaotus: **`SPIKKER.md` = kuidas asju käivitada.** **See fail = mida me
teame ja mis seisus oleme.** Arhiveeritud uurimused: `docs/arhiiv/`.

Viimati uuendatud: **28.08.2026**

---

## 1. Mis praegu tootmises jookseb

| teenus | port | mudel | mootor |
|---|---|---|---|
| `llama-server-print` | 8080 | `print-base-r64-20260827-Q8_0` | llama.cpp |
| `llama-server-hand` | 8081 | `kurrent-20260602-Q8_0` | llama.cpp |
| `ocr-service` | — | klient mõlemale | HTTP |

GPU 24,95 / 32,6 GB. Mõlemal serveril **`--image-max-tokens 5000`**, klient
teeb **`fit_to_grid` + PNG**. Kõik kolm on kohustuslikud — vt §2.1.

Lüliti: `ENGINE_CONFIGS` failis `kataloogi-jalgimine-ja-ocr.py`.
Trükiserveri unit-faili varukoopia:
`/etc/systemd/system/llama-server-print.service.bak-20260828`.

---

## 2. Kindlaks tehtud

### 2.1 llama.cpp on trükipoolel kasutuskõlblik — vana vastupidine järeldus on surnud

`clip.cpp: set_limit_image_tokens(8, 4096)` kärpis pilti **vaikides** ~8 %.
See, mitte mootor, kaotas marginaaliveeru. Ahelaga PNG + `--image-max-tokens
5000` + `fit_to_grid()`:

| 20 GT-holdout-lehte | transformers bf16 | llama.cpp Q8_0 |
|---|---|---|
| CER | 2,0 % | 1,9 % |
| `<m>` / 185 | 170 | 169 |
| kiirus | 30,8 s/lk | **6,2 s/lk** |

Kontroll, et kärbet pole: tekstipäring `prompt_tokens` = 825; pildiga 5777 →
**4952 visuaaltokenit**. Alla ~4100 tähendab, et lipp on puudu.

### 2.2 `<m>` ja `<i>` on kaks telge, mitte alternatiivid

`<m>` = roll ja asukoht. `<i>` = tüpograafia. Toores VUTT: **4 985 / 8 505
(58,6 %) marginaalidest on `<m><i>…</i></m>`**, sest need ON lehel kursiivis
(kontrollitud skaneeringult, Menii lk 0030). 41 teost on iseendaga vastuolus.

`strip_italics_in_marginalia()` võttis need maha → mudelile öeldi „marginaal
ei ole kunagi kursiiv", aga lehel ta on. Lipp:
`clean_markup(t, keep_marginalia_italics=True)`.

### 2.3 Menii vealiik: mudel LOEB ääreveeru, aga ei nimeta seda

10 lehel `1635-1 Frid. Menii`, kus uus mudel annab 0 `<m>`, on VUTT-i
marginaaliread väljundis **~87 % ulatuses olemas**:

| | ridu | `<i>` sees | sildita | päriselt puudu |
|---|---|---|---|---|
| GGUF | 227 | 77 | 119 | 31 |
| transformers | 255 | 102 | 124 | 29 |

Lehe kaupa on see puhas kas-või: leht on kas „`<i>`-leht" või „sildita leht",
segu ei ole. **See ei ole nägemisprobleem** (vana vealiik oli — tekst puudus
üldse). Menii ei ole treeningus (0 lehte); tema tihedus 24–39 `<m>` on
treeningkomplekti p95+ saba.

### 2.4 Noatera on lehepõhine ja pöördub mootorit vahetades mõlemat pidi

Sondi 13 Menii lehest on `<m>` mõlemal mootoril 3 lehel — **aga eri
lehtedel** (transformers võitis 0020, kaotas 0028). Holdoutil sama muster
(tf kaotas 1650-7, GGUF sai 12/12; 1637-1 vastupidi). **Kumbki mootor ei ole
„see õige"**; treeninguga sama ahel ei ole automaatselt parem.

### 2.5 Treeningkomplekti märgenditihedus

2 578 näidet (holdout väljas): mediaan **0** `<m>`, p90 17, p95 23, p99 31,
max 40. Lehti ≥24 `<m>`: 4,3 %. `<i>` : `<m>` = 22 151 : 8 320.
`data/lehekyljed/metadata_markup.csv` on **kogu toores 1500 lk**, millest
ainult 720 Gezeliuse lehte on märgendatud (12 493 sünteetilist `<i>`);
780 lehte on täiesti märgendita.

### 2.6 Tokenieelarve

Max järjend praegu **7 849 / 8 192**; `--keep-m-italics` viib selle 7 999-ni.
Halvim teoreetiline kombinatsioon 2 274 teksti + ~5 000 visuaali + 853
juhis/mall = 8 127. **Kärbet ei ole, `max_seq_length` ei vaja tõstmist.**

### 2.7 Mis on VALE varasemates märkmetes

- „llama.cpp mtmd ei näe marginaaliveergu" → §2.1, ümber lükatud.
- „`¬` jõudis 20260827 jooksu" → ei jõudnud; `clean_markup` kutsub
  `normalize_hyphenation`'i ja `train_markup.py` kutsub `clean_markup`'i.
  Need 8 971 esinemist on tooreses CSV-s.
- „Tokenivaru on 50" → tegelik 343 (ja 193 lipuga).
- „`print-base-r64` näeb marginaali paremini" → §2.3, nägemine ei ole telg.

---

## 3. `print-base-r64-20260827` — mida ta tõestas ja mida ei

**Võit on päris** (20 GT-holdout-lehte; need olid tõenäoliselt VANA mudeli
treeningus, ehk võrdlus on uue kahjuks kallutatud):

| | vana `markup-20260720` | uus |
|---|---|---|
| CER | 5,0 % | **2,0 %** |
| `<m>` / 185 | 75 | **170** |
| `<cs>` / 16 | 1 | **16** |

**Aga katse on neljakordselt confounded:** korraga muutus (a) baas vs 1. etapi
adapter, (b) r=16 → r=64, (c) +1 500 lk `data/lehekyljed`, (d) `clean_markup`
muudatused. „Suurem r on parem" **ei ole** sellest järeldatav.

Menii peal läks halvemaks: `<m>` 664 → 590 (141 loobivaba lehte), 10 lehel
kadus kogu veerg (vana GGUF: 3 lehel). Vealiik on siiski parem — §2.3.

---

## 4. Mõõteriistad

| skript | mida mõõdab | lõks |
|---|---|---|
| `scripts/eval_print.py` | 20 GT-lehte: CER, `cer_plain`, `<m>` recall/F1/sisu-CER | `--keep-m-italics` peab vastama treeningule, muidu vale GT |
| `scripts/menii_probe.py` | 13 Menii lehte, treeninguga sama ahel | **see on sihtmõõt**, mitte holdout |
| `scripts/reocr_vutt.py` | 147 VUTT „Toores" lehte | llama-server-only; GT-d EI OLE, ainult teine arvamus |
| `scripts/eval_kurrent.py` | 73 Kurrendi holdout-lehte | eraldi korpus |

**Holdout on küllastunud** — `<m>` juba 170/185. Trükipoole edasine
paranemine on mõõdetav ainult Menii peal.

**Nimekiri EI OLE külmutatud:** `reocr_vutt.py` valik on päring ja kasvab
(143 → 147). Ajaloolist võrreldavust see rikub.

---

## 5. Lahtised küsimused, järjekorras

1. **`--keep-m-italics` A/B** — ette valmistatud, ootab käivitamist. Üks
   muutuja, kontrollrühm on tootmises. Käsud: mälus
   `keep-m-italics-ab-20260828`.
2. **`<m>` märgendust juurde** — ainus päris allikas on VUTT-is märgendamine.
   Menii 58 „Toores" lehte on treeningust täiesti väljas. Inimtöö, mitte GPU.
3. **`train_on_responses_only` A/B** — praegu treenitakse 813-tokenist juhist
   kaasa; mediaanlehel on see ~67 % treenitud tokenitest.
4. **Tühjad/hõredad lehed** — trükikomplektis 0 näidet, juhis lubab
   `[tühi lehekülg]`. Kurrendi poolel juba tehtud.
5. **`finish_reason == "length"`** jääb kliendis kontrollimata → kärbitud leht
   kirjutatakse vaikselt tervena.
6. **`loop_detect.py` `LOOP_MIN_REPS = 3`** annab `D. D. D.` pühendusvormelile
   valehäire → korrektne leht saab `.err` ja kaob kasutajale.
7. **`reocr_vutt.py` transformersi backend + nimekirja külmutamine.**
8. **Kurrendi treening** lükkus edasi (28.08 ööl jooksis trükimudel).

Punktid 3–7 on `docs/treening-ja-inferentsi-koodi-ulevaade-20260828.md`-st;
sealt leiab põhjendused ja mõõtmised.
