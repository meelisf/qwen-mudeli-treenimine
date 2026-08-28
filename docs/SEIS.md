# Seis: mida me teame

**Elav dokument.** Uuenda seda, ära tee uut kuupäevaga faili — just
kuupäevaliste paralleeldokumentide kuhjumine tekitas 28.08 hommikul segaduse,
kus ma kordasin ühest failist järeldust, mille teine fail oli juba ümber
lükanud.

Tööjaotus: **`SPIKKER.md` = kuidas asju käivitada.** **See fail = mida me
teame ja mis seisus oleme.** Arhiveeritud uurimused: `docs/arhiiv/`.

Viimati uuendatud: **29.08.2026**

---

## 1. Mis praegu tootmises jookseb

> **29.08 seisuga on kõik kolm teenust MAHA** — peatatud 28.08 treeningu ajaks
> ja veel taastamata (nõuab sudo-t). Tabel kirjeldab konfiguratsiooni, mitte
> hetkeseisu. Uus mudel `print-base-r64-mi-vl-20260828` ootab aktiveerimist,
> vt §5.1.

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

**Lõks: lipp on KAHES kohas ja mõlemas on vaja.** `build_vutt_dataset.py`
kutsub `clean_markup`'i juba andmestikku ehitades. Kui seal lippu ei ole, on
`<i>` `<m>` seest CSV-st juba kadunud ja `train_markup.py --keep-m-italics` ei
saa neid tagasi tuua — katse peamine muutuja kaob vaikselt. Mõõdetud 28.08:
ilma liputa ehitatud CSV kaotab VUTT-i poolelt **4 851 `<i>`-d** (14 527 →
9 676). Lipp lisatud ka `build_vutt_dataset.py`-le ja märgitakse `SOURCE.txt`-i.
Vana `20260804` CSV oli ehitatud enne strippimise lisandumist, seega seal olid
nad alles ja `train_markup` otsustas.

### 2.3 Menii vealiik: mudel LOEB ääreveeru, aga ei nimeta seda

10 lehel `1635-1 Frid. Menii`, kus uus mudel annab 0 `<m>`, on VUTT-i
marginaaliread väljundis **~87 % ulatuses olemas**:

| | ridu | `<i>` sees | sildita | päriselt puudu |
|---|---|---|---|---|
| GGUF | 227 | 77 | 119 | 31 |
| transformers | 255 | 102 | 124 | 29 |

Lehe kaupa on see puhas kas-või: leht on kas „`<i>`-leht" või „sildita leht",
segu ei ole. **See ei ole nägemisprobleem** (vana vealiik oli — tekst puudus
üldse). Tema tihedus 24–39 `<m>` on treeningkomplekti p95+ saba.

Menii oli treeningus 0 lehte; **alates 28.08 andmestikust on seal 1 leht**
(`r_acad_dorp_1635_1_0006`, 23 `<m>`, 10 `<i>`). `menii_probe.py` kasutab
lehti 0020–0048, seega **sond on endiselt puhas**. Leht 0007 jäi välja: seal
on `[tühi lehekülg]` märgend KOOS tekstiga — VUTT-is parandada.

### 2.3.1 Märgendamata lehed OLID põhjus — katse 28.08 kinnitas

`--keep-m-italics --valitud-lehekyljed` jooks
(`models/qwen3.5-ocr-print-base-r64-mi-vl-20260828`, 1 793 näidet, 450 sammu,
train_loss 0,061) vastas §2.3 küsimusele **jah**.

| telg | vana (20260827) | uus (mi-vl) |
|---|---|---|
| Menii sond, `<m>` kokku 13 lehel | 70 | **207** |
| holdout `<m>` (GT 185) | 170 | **181** |
| holdout `<m>` sisu-CER | 13,1 % | **9,5 %** |
| holdout `cer_plain` | 0,9 % | 0,8 % |
| holdout `<cs>` (GT 16) | **16** | 11 |
| fraktuur, märke/lk (Becker 9–140) | 1 669 | 1 663 |

Menii 10 „kadunud" lehest **6 taastus täielikult** (0 → 19…33 `<m>`).
Ülejäänud 4 (0025, 0027, 0029, 0037) on **uus, kitsam vealiik**: mudel loeb
veeru rida-realt välja ja paneb iga rea `<i>`-sse, aga väline `<m>` jääb
panemata (ainult-`<i>` ridu: 28 / 29 / 19 / 10; vanal 12 / 0 / 0 / 0). See on
täpselt §2.2 telg — `<i>` võidab `<m>` üle.

Kaks lehte läksid alla: 0020 (12 → 10) ja 0024 (31 → 22, pikast nimeloendist
kaob 9 kirjet).

**Fraktuur ei kannatanud** 132 märgendamata Beckeri lehe väljajätmisest:
lehtedel 9–140 (vana mudel nägi neid treeningus, uus mitte) on lahknevus
keskm 1,3 %, `⸗` 177 → 174, loope 0 mõlemal.

CER-i pealkirjanumbrit (3,4 → 1,4 %) **ei tohi võtta puhta võiduna** — see
tuleb valdavalt muutujast endast (`<i>` `<m>` sees, 103 → 242 / 270), mille
eest vana mudelit karistatakse. Aus telg on `cer_plain`: muutumatu.

Täisraport: `docs/markup-katvus-20260828-tulemused.md`.

### 2.4 Noatera on lehepõhine ja pöördub mootorit vahetades mõlemat pidi

Sondi 13 Menii lehest on `<m>` mõlemal mootoril 3 lehel — **aga eri
lehtedel** (transformers võitis 0020, kaotas 0028). Holdoutil sama muster
(tf kaotas 1650-7, GGUF sai 12/12; 1637-1 vastupidi). **Kumbki mootor ei ole
„see õige"**; treeninguga sama ahel ei ole automaatselt parem.

### 2.5 Treeningkomplekti märgenditihedus

2 578 näidet (holdout väljas): mediaan **0** `<m>`, p90 17, p95 23, p99 31,
max 40. Lehti ≥24 `<m>`: 4,3 %. `<i>` : `<m>` = 22 151 : 8 320.
`data/lehekyljed/metadata_markup.csv` on **kogu toores 1500 lk**. Varasem
sõnastus („720 Gezeliuse lehte on märgendatud") liitis kaks eri teost;
mõõdetud 28.08 (kogu CSV-s on `<m>` arv **0**, kõigil 1500 lehel):

| grupp | lk | tag'idega | kreekarikkaid | `<i>` |
|---|---|---|---|---|
| Gezeliuse Lexicon 21–440 | 420 | 420 | 435 | 12 493 |
| Lexicon 2–20, 441–448 | 27 | 0 | — | 0 |
| Comenius Ianua 1–274 | 273 | 0 | 133 | 0 |
| Becker 1644 (fraktuur) | 140 | 0 | 1 | 0 |
| muu (disputatsioonid, prantsuse tragikomöödiad) | 640 | 0 | 33 | 0 |

**Ainult see 640 on vale signaal.** Ääremärkused ON neil lehtedel olemas ja
transkriptsioonis sildita — `data/lehekyljed/images/image_0014.jpg` algab
keset marginaaliviiteid (`p. Ceuſul. Huetiut de orit fabul. Rom. paſſim`).
Ianua ja Lexiconi ette valmistatud vahemik on ausad nullid: ei ääremärkusi
ega kursiivi lehel. Nemad kannavad ka kreeka — 553 kreekarikast lehte 602-st.

### 2.5.1 Becker 1644 on aus `<m>`-null, aga vale `<cs>`-null

R. Becker, *Linteum Exorcisticum oder Der Bantuch*, Riia 1644 — viis jutlust
nõiakunstist, terviklik 140 lk (`data/lehekyljed/images/00001–00140.jpg`).
Skaneeringult kontrollitud (00002, 00075, 00121): **üks veerg, täislaius,
ääremärkusi ei ole.** Tekstis on lühike viiterida 3 lehel 140-st.

Ta on ühtlasi **ainus tõsine fraktuuriallikas**: 138 lehte `⸗`-ga, 1310
esinemist; kogu `data/vutt` annab 64 lehte / 401. Lexicon, Ianua ja „muu" 0.

**Aga:** Becker vahetab pea igal lehel fraktuuri ja antiikva vahel
(ladinakeelsed lõigud — 00075-l viierealine `Astutia spirituum nefandorum…`,
00121-l ~5 lõiku, 00002-l kursiivne antiikva-pealkiri) ja transkriptsioonis
on need **sildita**. 140 lehte `<cs>`-nulli VUTT-i 193 `<cs>`-lehe vastu ≈
42 % lahjendus tagil, mille uus mudel just võitis (1 → 16/16, §3).

Kasutaja otsus 28.08 (üle vaadatud): **sildita Becker jääb VÄLJA.** Ta tuleb
ainult `data/vutt` kaudu, nii palju kui teda VUTT-is käsitsi märgendatud on
(praegu lk 1–8, 37 `<cs>`), ja kasvab sedamööda, kuidas märgendus edeneb.
**Hind, mida tuleb hommikul mõõta:** fraktuur kukub 138 lehelt 72-le (8 VUTT-i
Beckerit + 64 muud `⸗`-lehte), `⸗` 1 658-lt 399-le. Kui fraktuuri
transkriptsioon halveneb, on põhjus siin.

### 2.5.2 Sama leht kahes allikas, vastuoluliste siltidega

Becker on korraga mõlemas allikas. VUTT-is on **lk 1–8 juba „Valmis"**
(`1644-becker-linteum__00001–00008.jpg`, kokku 37 `<cs>` ja 1 `<i>`), samad
lehed on `data/lehekyljed`-is sildita. Tekst on 90–99,7 % identne — sama leht
kahe vastuolulise sildiga on halvim võimalik signaal, ja `20260827` jooksus
oli ta täpselt nii sees.

`--valitud-lehekyljed` viskab kogu sildita Beckeri välja, seega duplikaati
enam ei teki. `20260827` jooksus oli ta sees mõlemal kujul korraga.

**Sama viga on laiemalt.** Tekstipõhine duplikaadiotsing (28.08) leidis
`data/lehekyljed` ↔ `data/vutt` vahel **40 identse sisuga lehte**, kus VUTT-i
versioon on märgendatud ja 1. etapi oma mitte — ühel paaril `<m>` 23 vs 0.
Kõik 40 on „muu" grupis, mille `--valitud-lehekyljed` niikuinii välja jätab,
aga kui „muu" kunagi tagasi tuleb, tuleb see enne läbi puhastada. Beckeri 8
jäid selle otsingu võrgust välja (üksikud märgierinevused lõhkusid täpse
võrdluse) — seepärast on Beckerile eraldi reegel.

**Mõõtelünk:** holdouti 20 lehes on `⸗` kahel lehel, mõlemal üks tükk —
fraktuuri seal sisuliselt ei ole. Fraktuurikahju jääks praeguse hindamisega
nähtamatuks, olenemata sellest, kumba pidi otsustada.

### 2.5.3 `data/vutt` oli 24 päeva vana — uuesti ehitatud 28.08

`SOURCE.txt` osutas snapshot'ile `20260804T143956Z`, ehk **ka
`print-base-r64-20260827` treeniti 4. augusti VUTT-seisu peal.** Uuesti
ehitatud snapshot'ist `20260828T001502Z`: 1113 → **1120 lehte**, ükski ei
kadunud, ükski olemasolev ei muutunud, holdout 20/20 terve. Lisandus:

- 1 Menii leht (§2.3)
- 4 lehte `[tühi lehekülg]` märgendiga — **esimesed tühjad lehed trükikomplektis**,
  §5.4 osaliselt kaetud (3 × Gezeliuse Lexicon, 1 × 1795 Regiae Academiae)
- 2 muud Lexiconi lehte

**Backup on öine (03:15, seis 00:15).** Samal päeval VUTT-is tehtud töö EI OLE
snapshot'is — see tuleb alles järgmisel hommikul.

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
- „`loop_detect.py` annab `D. D. D.`-le valehäire" → **ei anna.** `is_looped()`
  viskab viimase sõna ära (`text.split()[:-1]`, sest loopinud väljund lõpeb
  tokenilaes keset sõna), nii et sabaks jääb `D. D.` = 2 kordust < 3. Mõõdetud
  28.08: `'...consectetur D. D. D.'` → `None`, `'...patrono suo D. D. D.'` →
  `None`; alles neljas kordus (`D. D. D. D.`) annab `(1, 3)`. Väide sündis
  `docs/arhiiv/llamacpp-juurdlus-20260827.md:112` teoreetilisest arutlusest,
  mis viimase sõna kärpimist ei arvestanud, ja rändas sealt kahte edasisse
  dokumenti. Päris probleem on §5.6.

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

Kasutaja hinnang tootmises (28.08, vibe check päris töövoos): uus mudel on
käegakatsutavalt parem kui vana. Ehk holdouti võit ei ole ainult holdouti oma.

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

1. **Uue mudeli aktiveerimine** — `print-base-r64-mi-vl-20260828` võitis
   kontrollrühma `<m>` teljel selgelt (§2.3.1), aga **ei ole veel tootmises**.
   Vaja: GGUF-i konversioon, `--image-max-tokens 5000` pariteedikontroll
   holdoutil, siis `ENGINE_CONFIGS` failis `kataloogi-jalgimine-ja-ocr.py`.
   Teenused seisavad praegu (treeningu ajaks peatatud) — vajavad sudo-ga
   käivitamist.
1b. **`<i>` võidab `<m>` üle** — Menii 0025/0027/0029/0037. Järelejäänud
   vealiik pärast 28.08 katset; kitsam ja täpsemini sihitav kui vana.
   Kandidaat järgmiseks katseks.
1c. **`<cs>` regressioon holdoutil** (16 → 11). 20 lehte on vähe — kas päris?

2. **`<m>` märgendust juurde** — ainus päris allikas on VUTT-is märgendamine.
   Menii 58 „Toores" lehte on treeningust täiesti väljas. Inimtöö, mitte GPU.
3. **`train_on_responses_only` A/B** — praegu treenitakse 813-tokenist juhist
   kaasa; mediaanlehel on see ~67 % treenitud tokenitest.
4. **Tühjad/hõredad lehed** — trükikomplektis 0 näidet, juhis lubab
   `[tühi lehekülg]`. Kurrendi poolel juba tehtud.
5. **`finish_reason == "length"`** jääb kliendis kontrollimata → kärbitud leht
   kirjutatakse vaikselt tervena.
6. **Loobituvastust on kaks koopiat ja need on lahku jooksnud.**
   `scripts/loop_detect.py`: `max_period=30, min_reps=3` (kalibreeritud 438
   Kurrendi väljundil, 0 valehäiret). Teenuse oma `LoopStopper`
   (`kataloogi-jalgimine-ja-ocr.py:421–422`): `max_period=20, min_reps=16` —
   vana piir, mis ei püüa mõõdetud 26-sõnalist loopi. HTTP-tee kutsub juba
   `loop_detect.is_looped()`-i (rida 599), aga transformersi-tee jookseb
   endiselt oma koopial. Lahendus: lülitada `LoopStopper` mooduli
   konstantidele. **NB:** `D. D. D.` valehäiret siin EI OLE, vt §2.7.
7. **`reocr_vutt.py` transformersi backend + nimekirja külmutamine.**
8. **Kurrendi treening** lükkus edasi (28.08 ööl jooksis trükimudel).

Punktid 3–7 on `docs/arhiiv/treening-ja-inferentsi-koodi-ulevaade-20260828.md`-st;
sealt leiab põhjendused ja mõõtmised.
