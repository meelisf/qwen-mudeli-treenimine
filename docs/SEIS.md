# Seis: mida me teame

**Elav dokument.** Uuenda seda, ära tee uut kuupäevaga faili — just
kuupäevaliste paralleeldokumentide kuhjumine tekitas 28.08 segaduse, kus
kordasin ühest failist järeldust, mille teine oli juba ümber lükanud.

Tööjaotus: **`SPIKKER.md` = kuidas asju käivitada.** **See fail = mida me
teame ja mis seisus oleme.** Lõpetatud uurimused: `docs/arhiiv/` — neid ei
uuendata ja osa järeldusi seal EI KEHTI.

Viimati uuendatud: **06.10.2026**

---

## 1. Mis praegu tootmises jookseb

> Trükipool `print-base-r64-mi-vl-20261006` alates 06.10 09:56 (koos makronijuhisega,
> §4), käsikirjapool `kurrent-20261002` alates 04.10.

| teenus | port | mudel | alates |
|---|---|---|---|
| `llama-server-print` | 8080 | **`print-base-r64-mi-vl-20261006-Q8_0`** | 06.10 |
| `llama-server-hand` | 8081 | **`kurrent-20261002-Q8_0`** | 04.10 |
| `ocr-service` | — | klient mõlemale, `ENGINE_CONFIGS` = llamacpp | — |

GPU 24,3 / 32,6 GB. Mõlemal serveril **`--image-max-tokens 5000`**, klient teeb
**`fit_to_grid` + PNG**. Kõik kolm on kohustuslikud — §2.1.

**Käivitamine pärast treeningut / reebooti (nõuab sudo-t):**

```bash
cd /home/mf/Dokumendid/LLM/qwen3.5
sudo cp systemd/llama-server-{print,hand}.service /etc/systemd/system/  # kui unit muutus
sudo systemctl daemon-reload
sudo systemctl start llama-server-print llama-server-hand
curl -s http://127.0.0.1:8080/health && curl -s http://127.0.0.1:8081/health
sudo systemctl start ocr-service
```

**Tagasiteed** (unit-failis `-m`/`--mmproj`, siis `daemon-reload` + `restart`):

| mudel | varukoopia |
|---|---|
| käsikiri → `kurrent-20260829` | GGUF kettal; muuda unit'is `-m`/`--mmproj` |
| trükk → `print-base-r64-mi-vl-20260828` | `/etc/systemd/system/llama-server-print.service.bak-20261006` + `git revert 8165e49` (makronijuhis) |
| trükk → `print-base-r64-20260827` | GGUF kettal; `.bak-20260828` on veel vanem (`markup-20260722`) |

**Serverit sondeerides pane `"chat_template_kwargs": {"enable_thinking": false}`
päringusse.** Ilma selleta läheb mudel mõtlemisrežiimi, põletab 4 096 tokenit ja
tagastab **tühja `content`-i** `finish_reason: "length"`-iga — näeb välja nagu
katkine mudel, aga on päringu viga. Kõik meie kliendid saadavad lipu; ad-hoc
curl/python sond ei saada.

### 1.1 Kettal hoitakse ainult seda, mida server loeb

Server loeb ainult `*-Q8_0.gguf` + `mmproj-*-F16.gguf`. **Vaheastmed**
(`models/merged/*-bf16/`, `models/gguf/*-BF16.gguf`) on kustutatavad — kogu ahel
adapterist Q8_0-ni võtab ~2,5 min. 29.08 koristati nii 103 GB (`models/` 160 →
56 GB). LoRA adapterid (`models/qwen3.5-ocr-*`, ~300 MB tk) on tõeallikas ja
jäävad alati.

02.10.2026: vaheastmed, kõik `checkpoints-*` ja `markup-20260722` GGUF tõsteti
`~/_kustutamiseks_20261002/` alla (kustutab kasutaja).
`models/checkpoints-*` (~9 GB) on treeningu jätkamispunktid, lõppadapterid on
neist eraldi; suuresti surnud kaal, kustutamine on eraldi otsus.

### 1.2 Masina seis: draiver ja CUDA (09.10.2026)

**Üks draiver, üks toolkit.** Ubuntu `nvidia-driver-580-open` **580.178.04**
(kernel `7.0.0-38`), CUDA toolkit **12.8** kaustas `/usr/local/cuda-12.8`
(`.bashrc` PATH, `ld.so.conf` järjekorras esimene). Seda kasutavad KÕIK:
llama.cpp (`CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`), treeningu venv
(torch `2.10.0+cu128`, unsloth 2026.5.8) ja ocr-service.

**Mis oli valesti.** Draiver ja 12.8 olid omal ajal paigaldatud NVIDIA CUDA
repost (RTX 5090 vajas alguses 570+ ja CUDA 12.8-t), kõrval Ubuntu
`nvidia-cuda-toolkit` 12.4 (omas `/usr/bin/nvcc`-d, keegi ei kasutanud).
Repo oli välja lülitatud, paketid jäid. Ubuntu 580.178 pakend jagab faile
teisiti: `apt upgrade` jättis NVIDIA kinni (kept back), `full-upgrade` kukkus
kaks korda —
- `libnvidia-opticalflow.so.1`: CUDA repos `libnvidia-compute`, Ubuntus
  `libnvidia-decode` → `--force-overwrite`;
- uus `libnvidia-gl-580` **Conflicts** `libnvidia-egl-gbm1` (CUDA repo) →
  `dpkg -r --force-depends libnvidia-egl-gbm1`, siis `dpkg -i --force-overwrite`.
- **`apt --fix-broken install` pakkus `nvidia-driver-580-open` EEMALDAMIST**
  (järgmine `autoremove` oleks viinud ka DKMS-i) — keelduda, paigaldada
  vahemälu `.deb`-id otse dpkg-ga.

**Koristatud 09.10:** 12.4 toolkit + runtime-teegid, `nvidia-settings`/
`libxnvctrl0` 610, vanad EGL/gpucomp/püsivara 159, `cuda-keyring`,
`cuda-ubuntu2404-x86_64.list.disabled`. Alles: 12.8 toolkit (44 paketti,
„local" — väljalülitatud repost, aga draiveriga konfliktita), `nvidia-modprobe`
610 (`apt-mark manual`; loob `/dev/nvidia-uvm`-i, Ubuntus asendust pole).
`rc`/`ic` konfijäägid (vanad kernelimoodulid, `nvidia-persistenced` 610) kahjutud.

**Persistence mode on nüüd VÄLJAS** — Ubuntu `nvidia-persistenced` käivitub
`--no-persistence-mode`-ga (CUDA repo oma lülitas sisse). Teenustele/treeningule
ükskõik (töötav protsess hoiab GPU lahti); tagasi = systemd drop-in.

**Järgmine samm: draiver 595-open** (`ubuntu-drivers devices` soovitab
`nvidia-driver-595-open`; saadaval ka 580/595/615 + server). 580 on pika toega
haru ja toetab CUDA 13.0-ni — torch `cu128` + llama.cpp 12.8 jaoks pole
595-st sisulist kasu (kiiruse määravad CUDA teegid, mitte draiveri haru).
Vahetama peaks (a) enne CUDA 13.x torch/llama.cpp peale minekut, (b) et püsida
Ubuntu soovitatud rajal. **Teha pärast Kurrendi v4 treeningut + hindamist, mitte
enne/ajal.** Käik: teenused maha → `sudo apt install nvidia-driver-595-open`
(peab eemaldama 580 metapaketi, MITTE midagi `cuda-*-12-8`) → reboot → kontroll:
`nvidia-smi` draiver = moodul, `loss-voimsuspiirid` (450 W, `no_turbo=1`),
torch GPU-arvutus, `import unsloth`, llama-teenused käivituvad + `/health`,
üks OCR-proov. Vana/uue mudeli holdout-võrdlus samal draiveril.

**Kontrollimata 09.10:** kas llama-teenused **käivituvad** puhastatud
süsteemiga (jooksid mällu laetud teekidega; `ldd` ahel terve). Esimene
käivitus pärast treeningut on see kontroll.

---

## 2. Kindlaks tehtud

### 2.1 llama.cpp pildipiir oli kogu segaduse põhjus

`clip.cpp: set_limit_image_tokens(8, 4096)` kärpis pilti **vaikides** ~8 % ja
kaotas marginaaliveeru. See, mitte mootor. Ahelaga PNG + `--image-max-tokens
5000` + `fit_to_grid()` on GGUF transformersiga pariteedis **mõlemal poolel**:

| | CER | muu | kiirus |
|---|---|---|---|
| trükk, 20 GT-lehte | 1,9 % vs 2,0 % | `<m>` 169 vs 170 | 6,2 vs 30,8 s/lk |
| käsikiri, 73 lehte | 4,1 % vs 4,1 % (mediaan) | — | 4,5 vs 20,8 s/lk |

**Kontroll, et kärbet pole:** tekstipäringu `prompt_tokens` lahutada pildiga
päringu omast. Peab tulema ~pikslid/1024, meil ~4 950. Alla ~4 100 = lipp puudub.
Mõõdetud 31.08 tootmisserveritel: 4 930 (print) ja 4 932 (hand). Täisjuurdlus:
`docs/arhiiv/llamacpp-juurdlus-20260827.md` (**selle põhijäreldus on surnud**).

### 2.2 `<m>` ja `<i>` on kaks telge, mitte alternatiivid

`<m>` = roll ja asukoht. `<i>` = tüpograafia. Toores VUTT: **4 985 / 8 505
(58,6 %) marginaalidest on `<m><i>…</i></m>`**, sest need ON lehel kursiivis.
41 teost on iseendaga vastuolus.

`strip_italics_in_marginalia()` võttis need maha → mudelile öeldi „marginaal ei
ole kunagi kursiiv", aga lehel ta on. Lipp: `keep_marginalia_italics=True`.

**Lõks: lipp on KAHES kohas ja mõlemas on vaja.** `build_vutt_dataset.py` kutsub
`clean_markup`'i juba andmestikku ehitades — kui seal lippu ei ole, on `<i>`
CSV-st juba kadunud ja `train_markup.py --keep-m-italics` ei saa neid tagasi.
Mõõdetud: ilma liputa kaob **4 851 `<i>`** (14 527 → 9 676).

### 2.3 Menii vealiik ja märgenduskatvuse katse

Vealiik: mudel **loeb** ääreveeru, aga ei nimeta seda — marginaaliread on
väljundis ~87 % ulatuses olemas, kas `<i>` sees või sildita. **See ei ole
nägemisprobleem.** Lehe kaupa on see puhas kas-või.

Katse 28.08 (`--keep-m-italics --valitud-lehekyljed`, 1 793 näidet) vastas
küsimusele „kas märgendamata lehed õpetasid `<m>` ära jätma?" — **jah**:

| telg | vana (20260827) | uus (mi-vl) |
|---|---|---|
| Menii sond, `<m>` 13 lehel | 70 | **207** |
| holdout `<m>` (GT 185) | 170 | **181** |
| holdout `<m>` sisu-CER | 13,1 % | **9,5 %** |
| holdout `cer_plain` | 0,9 % | 0,8 % |
| fraktuur, märke/lk | 1 669 | 1 663 |

Menii 10 „kadunud" lehest **6 taastus täielikult**. Tootmisahelas 98 → 205 `<m>`.
Lai A/B 147 lehel: `<m>` 590 → 635. **Fraktuur ei kannatanud** 132 märgendamata
Beckeri lehe väljajätmisest.

CER-i pealkirjanumbrit (3,4 → 1,4 %) **ei tohi võtta puhta võiduna** — see tuleb
valdavalt muutujast endast. Aus telg on `cer_plain`: muutumatu.
Täisraport: `docs/arhiiv/markup-katvus-20260828-tulemused.md`.

### 2.4 Noatera on lehepõhine ja pöördub mootorit vahetades mõlemat pidi

Trükipoolel: `<m>` on mõlemal mootoril 3 lehel, **aga eri lehtedel**.
Käsikirjapoolel sama, tugevamalt: leht `16590_senatsp_UAT_047_19_017` loobib
transformersil (ratio 4,88) ja on Q8_0-l korras; leht
`15116_trolldomskommiss…` loobib **kõigis viies jooksus**, mõlemas mudelis,
mõlemal mootoril.

**Kumbki mootor ei ole „see õige"**, ja treeninguga sama ahel ei ole
automaatselt parem. Loobid on **lehe**, mitte mudeli omadus — vt §3.

### 2.5 Treeningkomplekti märgenditihedus (trükk)

2 578 näidet: mediaan **0** `<m>`, p90 17, p95 23, p99 31, max 40.
`<i>` : `<m>` = 22 151 : 8 320. `data/lehekyljed/metadata_markup.csv` on kogu
toores 1500 lk ja seal on `<m>` arv **0**:

| grupp | lk | tag'idega | `<i>` |
|---|---|---|---|
| Gezeliuse Lexicon 21–440 | 420 | 420 | 12 493 |
| Comenius Ianua 1–274 | 273 | 0 | 0 |
| Becker 1644 (fraktuur) | 140 | 0 | 0 |
| muu (disputatsioonid jm) | 640 | 0 | 0 |

**Ainult see 640 on vale signaal** — ääremärkused ON neil lehtedel olemas ja
transkriptsioonis sildita. Lexicon ja Ianua on ausad nullid.

**Becker 1644** on aus `<m>`-null (üks veerg, ääremärkusi ei ole), aga **vale
`<cs>`-null**: ta vahetab pea igal lehel fraktuuri ja antiikva vahel, sildita.
Ta on ühtlasi ainus tõsine fraktuuriallikas (138 lehte `⸗`-ga). Kasutaja otsus
28.08: **sildita Becker jääb VÄLJA**, ta tuleb ainult `data/vutt` kaudu.

**Sama leht kahes allikas** on halvim signaal: duplikaadiotsing leidis
`data/lehekyljed` ↔ `data/vutt` vahel **40 identset lehte**, kus VUTT-i versioon
on märgendatud ja 1. etapi oma mitte (ühel paaril `<m>` 23 vs 0). Kõik 40 on
„muu" grupis; kui „muu" kunagi tagasi tuleb, tuleb see enne puhastada.

`data/vutt` ehitatakse VUTT-i **öisest** backupist (03:15, seis 00:15) — samal
päeval tehtud töö on snapshot'is alles järgmisel hommikul. `SOURCE.txt` hoiab
kuupäeva; 28.08 avastati, et CSV oli 24 päeva vana.

### 2.6 Tokenieelarve

Max järjend **7 999 / 8 192** (`--keep-m-italics`-iga). Halvim teoreetiline
kombinatsioon 8 127. **Kärbet ei ole, `max_seq_length` ei vaja tõstmist.**

### 2.7 Mis on VALE varasemates märkmetes

- „llama.cpp mtmd ei näe marginaaliveergu" → §2.1, ümber lükatud.
- „trükipoolel GGUF-i ei kasuta" → §2.1, ümber lükatud.
- „`loop_detect.py` annab `D. D. D.`-le valehäire" → **ei anna.** `is_looped()`
  viskab viimase sõna ära, sabaks jääb `D. D.` = 2 kordust < 3. Alles neljas
  kordus annab häire. Väide sündis teoreetilisest arutlusest ja rändas kahte
  dokumenti. Päris probleem on §6.4.
- „`<cs>` regressioon holdoutil (16 → 11)" → 147 lehe peal 166 → 160, müra.
- „`print-base-r64` näeb marginaali paremini" → §2.3, nägemine ei ole telg.
- „Teenuse juhise parandamine (`INSTRUCTION` → `KURRENT_INSTRUCTION`) on tasuta
  võit" → **CER-i võitu EI OLE** (mõõdetud 31.08, 0,1 pp). Juhis on siiski
  01.09 mudelipõhiseks tehtud — selguse, mitte täpsuse pärast, §3.3.
- „Uus Kurrendi mudel kaotab lühendi `u.`, see on regressioon" → **ei ole
  viga**, kasutaja otsus 31.08: lühend võib olla nii või naa, sisu on tähtis.
- „Tokenivaru on 50" → tegelik 343 (193 lipuga).

---

## 3. Kurrendi mudel `kurrent-20260829`

Treening 29.08 14:15 → 31.08 02:35 (**36,3 h**, 4244 sammu, 2 epohhi,
16 971 näidet, baas `unsloth/Qwen3.5-9B`, r=64, loss 0,132).
Täisnumbrid: `docs/kurrent-20260829-tulemused.md`.

### 3.1 Võit on selge — aga vaata mediaani, mitte keskmist

| jooks | med-CER (loopideta) |
|---|---|
| vana bf16 (20260602) | 6,5 % |
| **uus bf16** | **4,1 %** |
| **uus Q8_0** | **4,1 %** |
| uus 1. epohh | 4,9 % |

**Uue mudeli koond-CER bf16-l on 13,9 % — täpselt sama mis vanal. See on
juhus**, mitte tulemus: mõlemal veab sama üks loopinud leht keskmise üles.
Keskmine on paksusabaline, otsustav mõõdik on mediaan loopideta lehtedel.

**23 lehel, mida vana mudel treeningul ei näinud** (aaeb, hanse, dresdner_1665,
senatsprotokolle): mediaan **8,6 % → 4,0 %**. See on ausaim võrdlus — puuduvate
andmete parandus (20260602 vaikne piltide-puudumise viga, +4 259 lk) tasus end
ära.

**Teine epohh tasus end ära** — erinevalt 20260602-st: mediaan 4,9 % → 4,1 % ja
kadus katastroofiline loop (1. epohh läks ühel bullingeri lehel lõhki,
ratio 27,07). Jooksu poole lühemaks lõigata ei tasu.

### 3.2 Kaks asja, mida treening EI parandanud

**Loobid ei ole andmete puudus.** senatsprotokolle'i 229 lk lisamine parandas
allika mediaani 20,0 % → 7,2 %, aga just see üks leht, mille pärast neid muu
hulgas lisati, loobib transformersi teel edasi ja **läks hullemaks**. Õppimine
toimus; loop on lehe omadus (§2.4). Lahendus on kliendipoolne loobituvastus.

**`bullinger_autoren` +2,0 pp halvemaks** (9 lk, 8,1 % → 10,1 %) — ainus
tõsiseltvõetav regressioon. Hüpotees (KONTROLLIMATA): andmestikku ei
tasakaalustatud, rootsi ametkonnamaterjali mass kasvas. Enne midagi ette
võtta: vaata neid 9 lehte silmaga.

### 3.3 Teenuse juhis on nüüd sama, millega treeniti (01.09)

Kuni 01.09.2026 saatis teenus käsikirjamudelile `INSTRUCTION`-i, mitte
`KURRENT_INSTRUCTION`-it. **Mõõdetud 31.08, sama 73 lehte, sama server:**

| | CER | sõnaalguliste suurtähtede osakaal | `u.` osakaal |
|---|---|---|---|
| ground truth | — | 19,7 % | 2,0 % |
| `KURRENT_INSTRUCTION` | 7,6 % | 19,8 % | 3,3 % |
| `INSTRUCTION` (teenuse oma) | 7,5 % | 19,7 % | 3,3 % |

**CER-is eristamatud** — juhise vahetamine ei ole kvaliteedivõit ja seda ei
tohi sellisena esitleda. **Muudetud siiski 01.09 kasutaja otsusega, ja põhjus ei
ole CER:** juhis oli treeningus sees kogu aeg, seega on ta osa mudeli
sisendjaotusest. Teistsuguse juhisega päring on definitsiooni järgi jaotusest
väljas, ka siis kui 73 lehte seda juhuslikult välja ei too. Peale selle tekitasid
kaks lahknevat konfiguratsiooni (teenus üht juhist, `eval_kurrent.py` vaikimisi
teist) segadust: mõõdetud seis ei olnud kunagi see, mis tootmises jooksis.

Teenuses on nüüd `INSTRUCTIONS = {"print": INSTRUCTION, "hand":
KURRENT_INSTRUCTION}`; `get_instruction()` ja `get_chat_template()` võtavad
tüübi. **Nähtav muutus väljundis:** reavahetuse sidekriips `-` → `¬` (VUTT-i
käsikirjakokkulepe, `KURRENT_INSTRUCTION` p 3) ja käsikirjaväljundis ei ole enam
XML-märgendeid isegi juhuslikult — trükijuhis neid nõuab, Kurrendi oma keelab.
Sondeeritud `data/test/hand/1689-2_lk004.jpg` peal, mõlemad juhised sama serveri
vastu.

### 3.4 Holdout ei sisalda VUTT-i päris žanre

VUTT `crx9xb/1` („kirjad Kambjast", herrnhutlaste eestlastest vendade kiri)
võrdlus näitas, et **uus mudel loeb sisu selgelt paremini** — vana andis mõttetu
„mit meiner *hohen* darinnen leben", uus loeb „Herzen"; samuti dich/das,
Wunden/Munden, Gemeine/Gemeinde (7×), sie/Sir, den/die. Uue vead: „Edo" pro
„Lodo"; mõlemad jätsid kohanime „Cambij" transkribeerimata.

Aga: Kambja kirjad ja XVII saj Tartu ladina protokollid **ei ole korpuses
esindatud**. §3.1 numbrid kehtivad **arhiivikorpuste** kohta ja ei ütle VUTT-i
kasulikkuse kohta midagi. Enne järgmist treeningut lisa holdouti päris VUTT-i
materjali, muidu jääb iga järgmine jooks sama nurga taha.

Vana diagnoos on siin endiselt elus ja nüüd kinnitust saanud: **herrnhutlaste
lühendisüsteem ja eesti kohanimed saksa kujul** on mudelile võõrad. Vt
`docs/arhiiv/kurrent-strateegia.md` ja mälu `kurrent-korpuse-keeleline-auk`.

### 3.4b Andmestik v2 (02.10.2026) — järgmise treeningu sisend

`data/kurrent/` on ümber ehitatud (`scripts/build_kurrent_v2.py`), **18 908 rida**:
vana 17 044-st jäi 10 044 (Zürich 8 000 → 1 000), lisandus 7 571 dedup'itud
kurrent_xix mitte-Zürichi lehte (sh **2 566 XVIII saj saksa**, Greifswaldi
konsiilium) ja 1 292 `hanse_kurrent_xvii`. `vutt_horedad` 27 (värskeim VUTT).
Põhjus ja audit: `docs/kurrent-andmestikud.md` „Ülevaatus 2026-10-02".

- Uued allikanimed: `kurrent_xix_zurich` (endine `kurrent_xix`),
  `xix_read_1750_99/1800_49/1850_99/1900/dateerimata`, `hanse_kurrent_xvii`.
  Vanad allikata read (aaeb, jonkopings) said sildi. Täpne projekt:
  `data/kurrent/projektid.csv`.
- **Holdout 133** = vanad 73 MUUTMATA (vana↔uue mudeli võrdlus) + 10 iga uue
  allika kohta. Vana mudeli võrdluseks filtreeri eval vanade 73 peale.
- Vana andmestik ja ehitamise lähteandmed (HF cache) on
  `~/_kustutamiseks_20261002/` all — kustutab kasutaja.
- Treeninguks: `train_kurrent.py` ilma muudatusteta; ~11 % rohkem näiteid →
  ~40 h 2 epohhiga.

### 3.4c Treening 02.10.2026 — kolm muudatust retseptis

Võrreldes 29.08 jooksuga muutus lisaks andmestikule (§3.4b) kolm asja.
Puhast A/B-d ei tehtud, sest üks jooks võtab ~35–40 h:

1. **`train_on_responses_only`.** Juhis (375 tokenit, identne igas näites)
   oli 47 % kaotokenitest. Kontrollitud collatoriga: treenitav osa on nüüd
   `<think>\n\n</think>\n\n` + transkriptsioon + `<|im_end|>`.
   **Kadu ei ole 29.08 numbriga (0,132) võrreldav**, võrdle holdouti CER-i.
2. **Üle 8192 tokeni näited välja** (2 koenigsfelden_adhr lehte, kuni 15 754
   tokenit). Pikkuse hinnang on tegelikust +16 tokenit (kontrollitud 6 näitel).
3. **Baas bf16, mitte 4-bit** (`--16bit`). Mõõdetud `--test`-iga 20 PIKIMA
   näite peal (6 987–7 737 tokenit): 4-bit 19,7 GB / 160 s, **bf16 28,0 GB /
   129 s** (−19 %). Kaart 32 GB.
   **Päris jooksus kiirusevõitu EI OLE:** 32,7 s/samm vs 29.08 4-bit
   30,8 s/samm, ETA ~42 h. −19 % kehtis ainult pikimatel näidetel.
   nvidia-smi näitab jooksu ajal 30,9 / 32,6 GB — varu ~1,7 GB, OOM-i korral
   `--resume --16bit`.
   **Ainus põhjus** on treeningu ja inferentsi kooskõla: QLoRA adapter õpib
   osaliselt kompenseerima NF4 kvantiseerimisviga, mida tootmise bf16 → Q8_0
   baasis ei ole. Oodatav võit on VÄIKE (QLoRA artikkel: NF4 ≈ 16-bit LoRA).
   Ainus otsene vihje: testis sama 20 näite ja seemnega oli bf16 kadu igal
   5 sammul 0,07–0,15 madalam (1,66 vs 1,76) — 5 sammu, mitte tõend.
   Kui see jooks ei ole 29.08-st parem, ei tea me, kas süü on andmetel või
   siin — kolm muudatust korraga on teadlik kompromiss.

LoRA r=64/alpha=64 jäi: vead on tähekuju-lugemisvead, mitte mahupiirang
(arutelu 02.10).

**Holdouti kontroll (02.10, enne võrdlust loe):** vanad 73 on uues
`holdout.txt`-is alles, GT-tekst ja pilt baithaaval identsed vana andmestikuga
(`~/_kustutamiseks_20261002/data_kurrent_vana_17044/`). Vana mudeli väljundid:
`data/kurrent/eval/kurrent-20260829-Q8_0/`. Teksti duplikaate holdout↔treening
ei ole. **Pildi duplikaate on 3, kõik vanade 73 seas ja olid ka 29.08
treeningus** (leke oli mõlemas jooksus ühesugune):

- `11771_bullinger_au_1209460_0002_49174186.jpg` — sama pilt treeningus 6×
  (11741/46/50/60/66/76), eri transkriptsioonidega
- `16441_aaeb_xiv_xvi_3680035_0013_76516926.jpg` — 3× (15236, 16375, 16515)
- `15464_aaeb_xiv_xvi_1627450_0002_60945177.jpg` — 1× (16494)

**Aus võrdlus = 70 lehte**, need 3 mõlemast välja. Uued 60 on puhtad; vana
mudel tuleb nende peal eraldi läbi lasta.

**Kõrvalleid, `bullinger_autoren`:** sama pilt esineb eri ID-dega mitu korda ja
eri GT-ga, üks neist rämps-HTR (`11766`: „lo illis oii m coditit detit…").
Kandidaat §3.2 bullingeri regressiooni seletuseks — kontrollimata, kui
laialt see allikas levib.

### 3.4d Andmestik v4 (04.10.2026) — iga allikas auditeeritud

GT-kontroll (§6.11 c) leidis, et PAGE XML-i ehitajad jätavad **tühja
TextLine-i vaikselt vahele** → GT-st puuduvad pildil olevad read. Allikas:
CITlabi „Matcher" automaatjoondus — **kogu Zürich** (999/1 000), Escher,
semper, Pyl, parthey, hufeland, nn_msgermqu — ja Bullinger (lisaks mitu
XML-versiooni sama faili kohta). ≥ 2 tühja reaga lehtedest 58 % on
GT-kontrolli kandidaadid, 0–1-ga 3–5 %. GT-kontroll üksi seda ei püüa
(Zürich: CER 0,5 %, mudel õppis samad read vahele jätma) → kõik allikad
auditeeritud XML-i tasemel; Dresdneri vana TEI-ehitaja oli eraldi katki.

**`data/kurrent_v4/` = 18 973 rida** (treeninguks 18 843): v2 − osalise GT-ga
lehed (xix 1 684, Zürich 999, Bullinger, Trolldom 19, Königsfelden 3);
Bullinger 1 837 → 709 (parim XML-versioon); Dresdner 241 → 166 (uus ehitaja);
+2 463 DTA Kosmos-Nachschriften (9 kätt 1827–29), +707 Geusau reisipäevik 1740,
+386 Sandersi kirjad (DTA TEI täistekst); Escher 201 → 768.
**Holdout 133 → 130**: −23 katkist, +20 DTA (Geusau 10, Kosmos 10). Detailid, auditi tabel ja
skriptid: `docs/kurrent-andmestikud.md` „Ülevaatus 2026-10-04".

Parandus: ridade vahelejätmine EI tulnud alles v2-ga — 29.08 andmestikus oli
8 000 Zürichi Matcheri-lehte. Vanade 73 holdout-lehe võrdlus (29 vs 31 rida)
oli osaliselt vigase GT peal. Tagasivahetust pole vaja; järgmine treening v4-ga
(nädalavahetus 10.–11.10). Zürichi täistekst on olemas (Zenodo 10517999),
taastamine järgmises iteratsioonis.

**Kontrollnimekiri: kohe kui GT-kontroll on valmis (~05.10 17:00)** — sammud 1–6 TEHTUD 05.–06.10 (vt `kurrent-andmestikud.md` „GT-kontroll ja väljavõtt"); lahti ainult 7.

1. Jooks lõppenud? `tail logs/gt-kontroll-20261004-1535.log`
   (viimane rida `[18775/18775]`), siis `venv/bin/python scripts/gt_kontroll.py --aruanne`.
2. Kandidaadid allikate kaupa üle (`--naita <tüvi>`, paar näidet allika kohta):
   GT viga vs raske käsi. v2-st pärit lehti on mudel treeningus näinud →
   CER ≥ 10 % on tugev GT-vea märk, aga ära viska tervet allikat silmaga
   vaatamata välja.
3. **Vahetus** (enne seda EI tohi `data/kurrent`-i puutuda — jooks loeb seda):
   ```bash
   K=~/_kustutamiseks_20261005; mkdir -p $K
   mv data/kurrent $K/data_kurrent_v2          # vabaneb ~5 GB (v4-st välja jäänud lehed)
   mv data/kurrent_v4 data/kurrent
   # tööriistade väljundid kaasa — gt_kontroll jätkab tüve järgi, eval/oesel loevad siit
   mv $K/data_kurrent_v2/{gt_kontroll,eval,oesel} data/kurrent/
   rm -rf data/kurrent_v3 data/bullinger_v3 data/dta_kosmos data/dta_lisa data/escher_lisa data/dresdner_v2 data/dta_tei/img
   ```
   Vahekaustad on ainult CSV-d + hardlinkid (kokku ~40 MB) ja skriptidega
   uuesti ehitatavad. `data/dta_tei/*.xml` JÄÄB (TEI vahemälu, 9 MB).
4. **Uued lehed läbi sama mudeli:** `venv/bin/python scripts/gt_kontroll.py --paralleel 3`
   — jätkab tüve järgi, st teeb ainult `dta_*`, `esch_*`, `blv3_*`, `dresdner1665_*` (~5 000 lk, ~7 h).
   NB: gt_kontroll jätab holdout'i vahele → 20 DTA holdout-lehte vaata silmaga
   (`build_dta_kosmos.py --naita <id> <nr>` vs pilt).
   Siin sõelu AINULT struktuurselt (väljund GT-st > 1,3× pikem, loop, GT algab
   keset lehte) — mudel ei ole neid käsi näinud, kõrge CER ei tõenda GT viga.
5. Kandidaadid (2 + 4) välja `data/kurrent/metadata.csv`-st (skript on veel
   kirjutamata; holdout jääb puutumata), dokumenteeri `kurrent-andmestikud.md`-sse.
6. `venv/bin/python scripts/train_kurrent.py --test --16bit` — NB: `--test` võtab ALATI 20 pikimat
   näidet, rida `Andmestik:` näitab siis 20. Kontrolli, et `Hoiatus`/`Üle … tokeni` ridu ei ole ja
   `Holdout: 130`; täisandmestik = metadata.csv 18 846 − 130 = **18 716**. Jõudega OCR-teenus hoiab
   GPU-l ~26,5 GB → mudeli laadimine kukub `meta tensor`-iga; see on oodatav, testi treeningupäeval.
7. Treening v4-ga nädalavahetusel 10.–11.10 (retsept §3.5; OCR-teenus maas, teavita kasutajaid).
   **Enne `train_kurrent`-i** (teenus on juba maas): `scripts/prompt.py` `KURRENT_INSTRUCTION`-isse
   reegel „lühendusmärk (rõhtjoon tähe kohal, nasaal/geminatsioon) = makron U+0304, mitte tilde"
   — GT on v4-s juba makroniga (VUTT ADR 0062). **Valmis patch:** `git apply logs/prompt-kurrent-makron-v4.patch`
   (06.10; v4 kontrollitud: 18 846 rida, holdout 130, tilde 0, makron 5 135, NFC, pilte puudu 0).
   Patch muudab ka pika s-i rea: „ſ – transcribe as ſ" → „long s (ſ) – transcribe as round s".
   **Pikk s ühtlustatud 06.10:** ſ oli ainult 445 lehel (senatsprotokolle 229, dresdner_1665 165,
   xix_read_dateerimata 51), juhis käskis ſ-i, ülejäänud 18 400 lehte s-iga → `metadata.csv` ſ → s
   (varukoopia `metadata.csv.bak-pikk-s-20261006`), `build_kurrent_v4.py` teeb sama, `eval_kurrent.py`
   võrdsustab ſ ≡ s (holdoutis 7 ſ-lehte; vanad jooksud jäävad võrreldavaks). Kasutaja: kurrentkirjas
   on pikk s niikuinii reegel, VUTT-i toimetajad käsikirjas seda ei erista. Teenus laeb juhise käivitusel `prompt.py`-st →
   juhis ja mudel lähevad tootmisse KOOS (§3.3); varem muutes saaks vana mudel uue juhise.
   NB: trüki-`INSTRUCTION` ütleb veel „ũ, ñ, õ – keep as is (tilde preserved)" — see muutub
   trükimudeli järgmise treeninguga (VUTT #533 samm 4), mitte nüüd.
   `~/_kustutamiseks_20261002/hf_cache/` kustutada alles pärast treeningut.

### 3.5 Retsept, kui vaja korrata

`PYTHONUNBUFFERED=1 venv/bin/python scripts/train_kurrent.py --16bit` (alates 02.10; ilma PYTHONUNBUFFERED-ita jääb `| tee` logis loss puhvrisse; 29.08 jooks oli
ilma lippudeta, 4-bit) — muud vaikeväärtused
on õiged (baas `unsloth/Qwen3.5-9B`, r=64). `--test` peab näitama:
`Lähtepunkt: unsloth/Qwen3.5-9B`, `LoRA rank: 64`, `Holdout: 133`,
`Andmestik: 18773` + `Baasi laadimine: bf16` (andmestik v2, 02.10.2026, 2 liiga pikka välja; enne `Holdout: 73`, `Andmestik: 16971`). Kui ei näita — **peatu**.

Checkpointimine: `save_steps=250` (~2 h), `save_total_limit=3`, iga epohhi lõpus
rotatsioonist väljas `epohh-N-adapter/`. `--resume` leiab viimase checkpointi ise
ja **jätkab vana kuupäevatempliga** (jooks ületab südaöö).

Enne treeningut **kolm teenust maha** (`ocr-service llama-server-print
llama-server-hand`) — ainult `ocr-service` ei ole piisav, GPU-l on ka kaks
llama-serverit (~25 GB).

~~Kosmeetiline andmeviga (puuduv `allikas`-veerg)~~ — v4-s parandatud, kõigil 18 846 real 3 veergu (06.10).

---

## 4. Trükimudel `print-base-r64-mi-vl-20261006` (tootmises 06.10)

**Hindamine 06.10** (`logs/truki-v2-eval-20261006-0856.log`, skript `scripts/truki_v2_eval.sh`,
Q8_0 vana vs uus, kumbki oma juhisega): holdout `cer_plain` 0,8 = 0,8 %, `<m>` F1 0,81 → 0,85,
m_CER 9,8 → 8,1 %; Menii GGUF (`scripts/menii_gguf.py`) `<m>` 204 → 261, lehti 9 → 11/13
(0027/0029: 0 → 38/35); lühend makroniga (tildeid 0); Toores 205 lk loope 7 = 7.
**CER märgendusega 1,5 → 3,1 %** — kolm lehte, kus kursiivne põhitekst läheb `<i>`-sse, GT-s
mitte (pildil ongi kursiiv). v2-s valdavalt-`<i>`-põhitekstiga lehti 91 → 183 (43 → 69 teost):
VUTT-i konventsioon ebaühtlane, otsus kasutajal. Vahetus `scripts/truki_v2_tootmisse.sh`.

### 4.0 Eelmine: `print-base-r64-mi-vl-20260828`

Eelkäija `print-base-r64-20260827` võit oli päris (CER 5,0 → 2,0 %, `<m>`
75 → 170, `<cs>` 1 → 16), **aga katse oli neljakordselt confounded**: korraga
muutus (a) baas vs 1. etapi adapter, (b) r=16 → r=64, (c) +1 500 lk, (d)
`clean_markup` muudatused. „Suurem r on parem" **ei ole** sellest järeldatav.

Praeguse mudeli lisavõit tuli märgenduskatvusest (§2.3). Kasutaja hinnang
tootmises: käegakatsutavalt parem kui vana.

**Järelejäänud vealiik: `<i>` võidab `<m>` üle.** Menii lehtedel
0025/0027/0029/0037 loeb mudel veeru rida-realt välja ja paneb iga rea `<i>`-sse,
aga väline `<m>` jääb panemata. Sama neli lehte nullis **mõlemal mootoril**, ehk
see on mudeli, mitte ahela omadus. Kandidaat järgmiseks katseks.

**Treeningandmed on makroniga (05.10.2026, VUTT ADR 0062 / #533 samm 4).**
`data/lehekyljed` (mõlemad CSV-d) ja `data/vutt` läbisid `scripts/lyhend_makron_trukk.py`:
tilde → makron, prügi (U+E8BF → `q;`, U+F1A7 → `I`, kombineeriv märk rea alguses,
kreeka tilde → U+0342), kõik NFC. `build_vutt_dataset.py` teeb sama igal uuel ehitusel
(keelevalvur est/spa/por), `eval_print.py` võrdsustab tilde ja makroni.
**Järgmise trükitreeningu päeval** muuda `scripts/prompt.py` `INSTRUCTION`-is rida
„ũ, ñ, õ – keep as is (tilde preserved)" → lühendusmärk = makron — KOOS mudeliga (§3.3),
mitte varem. Detailid: `truki-andmestikud.md`.

**Järgmise trükitreeningu andmestik `data/vutt_v2` (05.10.2026)** — snapshot
`20261005T001501Z`, `--type print --keep-m-italics`: **1 488 lehte** (vana 1 120 + 368 uut,
vanad kõik alles, holdout 20/20 sama tekstiga). Uutel lehtedel on märgendus tihedam:
`<m>` 73 % lehtedest (vanadel 47 %), `<cs>` 25 % (17 %). Ehitaja rakendab
`lyhend_makron_trukk.puhasta`-t (makron + prügi), tildet/prügi ei ole.
Vahetus käsitsi treeningupäeval: `data/vutt` → `data/vutt_v1`, `vutt_v2` → `vutt`
(vii `eval/` ja `reocr/` kaasa — eval_print loeb `data/vutt/...`). Enne treeningut:
GT-kontroll 368 uuel lehel tootmismudeliga (GPU vaba pärast Kurrendi GT-kontrolli).

---

## 5. Mõõteriistad

| skript | mida mõõdab | lõks |
|---|---|---|
| `scripts/eval_print.py` | 20 GT-lehte: CER, `cer_plain`, `<m>` recall/F1 | `--keep-m-italics` peab vastama treeningule, muidu vale GT |
| `scripts/menii_probe.py` | 13 Menii lehte, treeninguga sama ahel | **see on trükipoole sihtmõõt**, mitte holdout |
| `scripts/reocr_vutt.py` | ~147 VUTT „Toores" lehte | GT-d EI OLE, ainult teine arvamus; nimekiri kasvab, ajalooline võrreldavus katki |
| `scripts/eval_kurrent.py` | 73 Kurrendi holdout-lehte | mõõdab arhiivitavasid, **mitte VUTT-i kasulikkust** (§3.4) |

**Mõlemad holdoutid on osaliselt pimedad.** Trükipoolel on ta küllastunud
(`<m>` 170/185) — edasine paranemine on mõõdetav ainult Menii peal.
Käsikirjapoolel ei sisalda ta VUTT-i žanre üldse.

**Iga jooks jätab tingimused faili** `<väljundikaust>/run.json` — kiirusnumbrit
ei tohi mälu järgi tsiteerida.

---

## 6. Lahtised küsimused, järjekorras

1. **Käsikirjamudeli katsetamine** — kasutaja testib paar päeva (alates 31.08).
   Tagasitee §1-s.
2. **`<i>` võidab `<m>` üle** (§4) — trükipoole järgmine katse.
3. **`<m>` märgendust juurde** — ainus päris allikas on VUTT-is märgendamine.
   Menii 58 „Toores" lehte on treeningust täiesti väljas. Inimtöö, mitte GPU.
4. **Holdouti laiendamine VUTT-i materjaliga** (§3.4) — muidu ei mõõda me seda,
   mida päriselt kasutame.
5. **Loobituvastust on kaks koopiat ja need on lahku jooksnud.**
   `scripts/loop_detect.py`: `max_period=30, min_reps=3` (kalibreeritud 438
   Kurrendi väljundil, 0 valehäiret). Teenuse `LoopStopper`
   (`kataloogi-jalgimine-ja-ocr.py:421–422`): `max_period=20, min_reps=16` —
   vana piir, mis ei püüa mõõdetud 26-sõnalist loopi. HTTP-tee kutsub juba
   `loop_detect.is_looped()`-i, transformersi-tee jookseb oma koopial.
   Lahendus: lülitada `LoopStopper` mooduli konstantidele.
   **Nüüd mõõdetud vajadus, mitte teoreetiline** (§3.2).
6. **`bullinger_autoren` regressioon** (§3.2) — 9 lehte silmaga üle vaadata.
   **04.10:** põhjus leitud — osaline GT (tühjad TextLine-id + vale XML-versioon), §3.4d.
7. **`train_on_responses_only` A/B** — praegu treenitakse 813-tokenist juhist
   kaasa; mediaanlehel ~67 % treenitud tokenitest.
   **PLAANIS ööl 06.→07.10** (kasutaja otsus 06.10): A = `print-base-r64-mi-vl-20261006`
   (treenitud 06.10 öösel, vana retsept, andmed `data/vutt` v2 + makron), B = SAMA käsk ja
   SAMAD andmed, ainult `train_markup.py` collatorisse `train_on_responses_only=True`,
   `instruction_part="<|im_start|>user\n"`, `response_part="<|im_start|>assistant\n"`
   (nagu `train_kurrent.py` 325–327). Ainus muutuja = mask. B vajab eraldi nime (muidu
   sama `DATE_STAMP`-i loogika annab uue kuupäeva, aga `_LIIK` sama — lisa nt `-ro`).
   Makronireegel juhises: `logs/prompt-makron-20261006.patch` (B-l sama juhis kui A-l).
   Võrdle holdouti CER/`cer_plain`/`<m>` ja Menii sondi, mitte kadu.
8. **Tühjad/hõredad lehed** — trükikomplektis 0 näidet, juhis lubab
   `[tühi lehekülg]`. Kurrendi poolel juba tehtud.
9. **`finish_reason == "length"`** jääb kliendis kontrollimata → kärbitud leht
   kirjutatakse vaikselt tervena.
10. **`reocr_vutt.py` transformersi backend + nimekirja külmutamine.**

11. **Treeningmaterjali kvaliteedikontroll (Kurrent).** Leitud 02.10: sama pilt
    eri ID-de ja eri GT-ga, osa GT-st rämps-HTR (`bullinger_autoren`, vt §3.4c
    „Holdouti kontroll"). Automaatselt saab leida KANDIDAADID, otsus jääb
    inimesele: (a) pildiräsi → vastuolulised GT-d (odav, kohe tehtav);
    (b) sõnade kehtivus / tähe-n-grammi skoor → rämps-GT (odav, CPU);
    (c) mudel üle treeningkomplekti, kõrge CER GT vastu = raske leht VÕI vale
    GT → ülevaatusjärjekord (~8 h GPU, alles pärast treeningut).
    **Kasutaja valik 02.10: tee (c)** — eesmärk ei ole ideaalne GT, vaid
    vigaste ja täiesti kontrollimata lehtede väljafiltreerimine.
    **04.10:** jooks käib (`scripts/gt_kontroll.py`, ~25 h, lõpp ~05.10 17:00).
    Vahetulemusest leitud süsteemne viga → §3.4d, andmestik v4. Pärast jooksu:
    allikate kaupa näidised üle, siis kandidaadid v4-st välja.

12. **Masina koristus pärast Kurrendi v4 treeningut** (09.10 kokku lepitud):
    (a) `~/_kustutamiseks_20261002/hf_cache/` (85 GB, toor-XML) — alles PÄRAST
    treeningut; (b) vahemälud `~/.cache/pip` 25 GB, `~/.cache/uv` 16 GB,
    `~/.cache/huggingface` muu kui `models--unsloth--Qwen3.5-9B` (baasmudel,
    JÄÄB); (c) kasutaja otsustab: `qwen-treening`, `~/kraken-test`, `~/.pyenv`,
    väikesed `ERR-rel`, `pagexml-to-alto`, `jaanson`, `trocr`, `AUTO-OCR`,
    `gemini-cli`, `~/.paddlex`; (d) draiver 595-open (§1.2).
    Tehtud 09.10: `disp`, `tartu-acad`, `vanad - VUTT…`, `HTRflow-riksarkivet`,
    `EstLLM-finetune` kustutatud (279 GB); CUDA koristus §1.2.

Punktid 5, 7, 9, 10 pärinevad
`docs/arhiiv/treening-ja-inferentsi-koodi-ulevaade-20260828.md`-st.
