# Kurrendi mudel `qwen3.5-ocr-kurrent-20260829` — tulemused

Treening 29.08 14:15 → 31.08 02:35, **36,3 h** (2176 min), 4244 sammu,
2 epohhi, 16 971 näidet, LoRA r=64, baas `unsloth/Qwen3.5-9B`.
Lõpp-loss 0,132. Järelahel (GGUF + 3 hindamisjooksu) lõppes 03:36.

**Kokkuvõte: uus mudel võidab selgelt.** Puhaste lehtede mediaan-CER
**6,5 % → 4,1 %** (−37 % suhtelist). Võit on suurim just seal, kus ta pidi
olema: lehtedel, mida vana mudel treeningul ei näinud.

Hindamine: `data/kurrent/holdout.txt`, 73 lehte, `KURRENT_INSTRUCTION`,
batch 4, pildieelarve 5,12M px, LANCZOS, 450 W.

## 1. Koondtulemus

Keskmine on paksusabaline — üks loopinud leht nihutab seda kümneid punkte.
**Otsustav veerg on „ilma loopideta" (ratio ≤ 1,4).**

| jooks | kõik 73: keskm | med | loope | ilma loopideta: keskm | **med** |
|---|---|---|---|---|---|
| vana bf16 (20260602) | 13,9 % | 6,6 % | 2 | 9,3 % | **6,5 %** |
| vana Q8_0 (27.08) | 15,6 % | 6,6 % | 3 | 9,2 % | **6,5 %** |
| **UUS bf16** | 13,9 % | 4,2 % | 3 | 6,6 % | **4,1 %** |
| **UUS Q8_0** | **7,6 %** | 4,2 % | 2 | 6,5 % | **4,1 %** |
| UUS 1. epohhi adapter | 44,5 % | 5,0 % | 3 | 7,3 % | 4,9 % |

**Uue mudeli koond-CER 13,9 % bf16-l on juhus, mitte tulemus** — see langeb
kokku vana mudeli numbriga puhtjuhuslikult, sest mõlemal veab ühe ja sama
lehe loop keskmise üles. Mediaan näitab tegelikku vahet: 6,6 % → 4,2 %.

## 2. Aus võrdlus: 23 lehte, mida vana mudel EI näinud

aaeb, hanse_kurrent_xvi, dresdner_1665, senatsprotokolle — need jäid
20260602-st välja (osalt vaikselt, puuduvate piltide tõttu).

| jooks | keskm (ilma loopideta) | med |
|---|---|---|
| vana bf16 | 10,6 % | 8,6 % |
| vana Q8_0 | 11,2 % | 8,5 % |
| **UUS bf16** | **6,1 %** | **4,5 %** |
| **UUS Q8_0** | **6,0 %** | **4,0 %** |

**Mediaan poolestus, 8,6 % → 4,0 %.** Puuduvate andmete parandus tasus end ära.

50 lehel, mis olid juba vana mudeli treeningus, on võit väiksem, aga olemas:
mediaan 6,4 % → 4,0 %.

## 3. Allika kaupa (mediaan-CER, loopideta)

| allikas | lk | vana | uus bf16 | uus Q8_0 | vahe |
|---|---|---|---|---|---|
| senatsprotokolle ᴜᴜꜱ | 3 | 20,0 % | 6,9 % | 7,2 % | **−12,8 pp** |
| dresdner_1665 ᴜᴜꜱ | 3 | 28,2 % | 17,6 % | 17,9 % | **−10,2 pp** |
| gota_hovratt | 3 | 14,5 % | 6,5 % | 6,6 % | −7,9 pp |
| hanse_kurrent_xvi ᴜᴜꜱ | 7 | 10,3 % | 2,5 % | 2,6 % | −7,7 pp |
| krigshovrattens | 3 | 12,6 % | 4,1 % | 4,9 % | −7,7 pp |
| bergskollegium_adv | 3 | 14,1 % | 6,8 % | 6,6 % | −7,5 pp |
| bergskollegium_rel | 9 | 7,2 % | 1,7 % | 1,5 % | −5,7 pp |
| trolldomskommissionen | 5 | 8,9 % | 3,9 % | 4,6 % | −4,3 pp |
| kurrent_xix | 10 | 0,6 % | 0,8 % | 0,8 % | +0,2 pp |
| svea_hovratt | 5 | 2,5 % | 3,4 % | 2,9 % | +0,3 pp |
| koenigsfelden_adhr | 3 | 5,1 % | 6,0 % | 6,0 % | +0,9 pp |
| aaeb (sildita) ᴜᴜꜱ | 10 | 2,8 % | 4,0 % | 3,8 % | +1,0 pp |
| **bullinger_autoren** | 9 | 8,1 % | 9,5 % | 10,1 % | **+2,0 pp** |

Kaheksa allikat paranesid tuntavalt, neli on müra piires.
**Ainus tõsiseltvõetav regressioon on `bullinger_autoren` (+2,0 pp, 9 lk)** —
XVI saj Šveitsi kirjad. Kandidaatseletus: andmestikku ei tasakaalustatud
(`metadata.csv`, mitte `metadata_balanced.csv`), ja rootsi ametkonnamaterjali
mass kasvas. **Kontrollimata hüpotees** — enne kui midagi ette võtta, tuleb
neid 9 lehte silmaga vaadata.

Ka `aaeb` (+1,0 pp) on veider: ta ON nüüd treeningus, aga läks pisut
halvemaks. 10 lehe mediaan, tõenäoliselt müra.

## 4. Q8_0 on bf16-ga pariteedis — nüüd korralikult mõõdetud

See mõõtmine oli võlgu: 27.08 Kurrendi mootorivõrdlus tehti **ilma**
`--image-max-tokens 5000` liputa, ehk pilt kärbiti vaikselt 4096 tokenini.

| | 73 lk | s/lk | lehte/tunnis | med-CER (loopideta) |
|---|---|---|---|---|
| transformers bf16, batch 4 | 25,3 min | 20,8 | 173 | 4,1 % |
| **llama.cpp Q8_0, `-np 4`** | **5,5 min** | **4,5** | **801** | **4,1 %** |

**4,6x kiirem, mediaan identne** (6,5 % vs 6,6 % keskm). Kvantimine ei maksa
midagi. GGUF: `models/gguf/kurrent-20260829-Q8_0.gguf` (9,07 GiB) +
`mmproj-kurrent-20260829-F16.gguf`, 427/427 tensorit, `--no-nextn` toimis
(viimane plokk `blk.31`, block_count 32).

## 5. Loopid — mudel neid ära ei parandanud, mootor otsustab

| leht | vana bf16 | vana Q8_0 | uus bf16 | uus Q8_0 |
|---|---|---|---|---|
| `16590_senatsp_UAT_047_19_017` | **2,93** | ok | **4,88** | ok |
| `15116_trolldomskommiss…` | **1,58** | **1,55** | **1,46** | **1,47** |
| `13220_bergskollegium_r…` | ok | **1,41** | **1,41** | **1,41** |
| `12740_bergskollegium_adv…` | ok | **4,52** | ok | ok |

(arvud = `ratio`, ehk väljundi pikkus / GT pikkus)

**Senatsprotokolle'i 229 lehe lisamine EI parandanud seda üht lehte, mille
pärast neid muu hulgas lisati.** Transformersi teel läks ta isegi hullemaks
(ratio 2,93 → 4,88, CER 286 % → 465 %); Q8_0-l teeb mudel selle lehe õigesti,
täpselt nagu vana mudelgi. Ülejäänud allika kolm lehte paranesid küll
korralikult (mediaan 20,0 % → 7,2 %), ehk **õppimine toimus, loop lihtsalt ei
ole andmete puudus.** See on Tübingeni senati kohalolijate nimekiri —
lühikesed korduvad read, struktuurselt loobiohtlik.

`15116_trolldomskommiss…` loobib **kõigis viies jooksus**, mõlemal mootoril,
mõlemas mudelis. See ei ole mudeli omadus, vaid lehe oma.

Kinnitab varasemat järeldust (SEIS §2.4): **noatera on lehepõhine ja pöördub
mootorit vahetades mõlemat pidi.** Lahendus ei ole treeningandmetes, vaid
kliendipoolses loobituvastuses (`loop_detect.is_looped()`).

## 6. Teine epohh tasus end ära — erinevalt 20260602-st

| | med-CER (loopideta) | keskm | loope |
|---|---|---|---|
| 1. epohh (samm 2122) | 4,9 % | 7,3 % | 3 |
| 2. epohh (samm 4244) | **4,1 %** | **6,6 %** | 3 |

−0,8 pp mediaani ja −0,7 pp keskmist. Lisaks **kadus üks katastroofiline
loop**: 1. epohhi adapter läks lehel `10272_bullinger_au_1193903_0096` täiesti
lõhki (ratio **27,07**, CER 2624 % — genereeris tokenilaeni), 2. epohhil on
see leht korras. 20260602 puhul andis teine epohh vähe; **seekord andis**, ehk
jooksu poole lühemaks lõigata ei tasu.

## 7. Mis jääb inimesele

1. **Aktiveerimine** — `llama-server-hand.service` tee uuele GGUF-ile +
   `MODEL_CONFIGS["hand"]`. Nõuab sudot.
2. **Kolm teenust on 29.08-st maas** (`ocr-service`, `llama-server-print`,
   `llama-server-hand`) — GPU oli treeningu all. VUTT-i OCR ei tööta enne,
   kui need üles pannakse.
3. **Silmaga vaatamine** — CER mõõdab vastavust arhiivikorpuse tavadele,
   mitte kasulikkust VUTT-is. Eriti: bullingeri 9 lehte (regressioon).
4. **Vahefailid** ~35 GB: `models/merged/qwen3.5-ocr-kurrent-20260829-bf16`
   + `models/gguf/kurrent-20260829-BF16.gguf`. Taastuvad adapterist ~2,5 min.
5. **`data/kurrent/metadata.csv`** — 2039 real puudub veerg `allikas`.
   Nüüd ohutu parandada, treening on läbi.

Väljundid: `data/kurrent/eval/{kurrent-20260829-Q8_0,qwen3.5-ocr-kurrent-20260829,kurrent-20260829-epohh1}/`
(iga kaust: lehekaupa `.txt`, `results.csv`, `run.json`).
Ahela kirjeldus ja logid: `docs/kurrent-20260829-oine-ahel.md`.

---

## 8. Järelmõõtmine 31.08: holdout on pime VUTT-i päris materjalile

Ajend: kasutaja võrdles VUTT-i lehte `crx9xb/1` („kirjad Kambjast",
herrnhutlaste eestlastest vendade kiri) vana ja uue mudeliga.

**Uus mudel loeb sõnu paremini** — vana andis „mit meiner *hohen* darinnen
leben" ja „Mein *hohes* verlangt", uus loeb õigesti „Herzen"/„Herze"; samuti
„dich" (mitte „das"), „Wunden" (mitte „Munden"), „Gemeine" (mitte
„Gemeinde", 7×), „sie" (mitte „Sir"). **Aga kaotab transkriptsioonitava:**
käsikirjas on läbivalt `u.` (15×), uus mudel kirjutab kõigil 15 korral
„und" — mõlema juhisega, ehk see on mudel, mitte juhis. Uus eksis ka
pealkirjas („Edo" pro „Lodo", kasutaja kinnitatud).

Põhjus on treeningkorpuses, mitte mudelis:

| allikas | lehti | `u.` | `und` |
|---|---|---|---|
| kurrent_xix (47 % korpusest) | 8000 | 327 | 46 297 |
| senatsprotokolle | 229 | 235 | 448 |
| **KOKKU** | **17 045** | **570** | **50 870** (98,9 %) |

### Juhise parandamine EI ole põhjendatud — mõõdetud

`eval_kurrent.py --prompt print` vs vaikimisi, sama 73 lehte, sama server:

| | CER | sõnaalguliste suurtähtede osakaal | `u.` osakaal |
|---|---|---|---|
| ground truth | — | 19,7 % | 2,0 % |
| `KURRENT_INSTRUCTION` | 7,6 % | 19,8 % | 3,3 % |
| `INSTRUCTION` (teenuse oma) | **7,5 %** | **19,7 %** | 3,3 % |

**Juhised on holdoutil eristamatud.** Kambja lehe suur vahe (14 vs 2
suurtähelist nimisõna) on n=1 ja väljaspool domeeni. `get_instruction()`
jääb muutmata — mitte enam „et mootorivahetus jääks ainsaks muutujaks",
vaid **sest muutmiseks pole tõendit**.

### Mida see holdouti kohta ütleb

Holdoutil **ei ole ühtki lehte, kus `u.` domineeriks** (esineb 20 lehel 73-st,
üheski mitte ülekaalus). Kambja leht on 100 % `u.`, mudel andis 0 % — ja
holdout ei suuda seda viga näidata. Sama kehtib XVII saj ladina Kurrendi
kohta (vt `kurrent-korpuse-keeleline-auk`).

**Järeldus: §1–3 numbrid on ausad arhiivikorpuste kohta ja ei ütle midagi
selle kohta, mida VUTT päriselt teeb.** Enne järgmist treeningut tuleks
holdouti lisada VUTT-i päris materjali — vähemalt Kambja-tüüpi kirjad ja
XVII saj Tartu protokollid —, muidu jääb iga järgmine jooks samasse
pimenurka. `u.` ise laheneb töövoos: leht on „Toores", inimene parandab,
järgmine treening näeb lühendit.
