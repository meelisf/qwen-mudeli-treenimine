# Plaan: trükipoole järgmine treening

Koostatud 27.08.2026, llama.cpp juurdluse järelmina
(`docs/llamacpp-juurdlus-20260827.md`). **v2 samal õhtul** — välise
ülevaatuse järel; parandused on all punktis „Mis v1-s valesti oli".
Ei puuduta reedest Kurrendi-jooksu — see on esimene järjekorras ja
unslothi/transformersit ei tohi enne liigutada.

---

## 0. Piiritlus: mida juurdlus tegelikult mõõtis

Kogu 27.08 juurdlus jooksis **llama.cpp mootoril**. Trükipoolel me seda
mootorit **ei kasuta** (otsus: `llamacpp_image_token_cap` / juurdluse lõpp).
Seega enamik juurdluse „katkist" ei ole trükimudeli viga:

| leht | teenus (transformers) | llama.cpp |
|---|---|---|
| `1635_1_0017` `<m>` | **28** | **0** |
| `1635_1_0036` `<m>` | 7 | 0 |
| `<cs>` 135 lehel | **131** | 57 |

**Tootmisahel loeb Menii ääreveeru välja.** llama.cpp kaotab selle. Iga
plaanipunkt, mis oli üles ehitatud „marginaalid kaovad" peale, tuleb seega
üle vaadata — allpool on see tehtud.

Mis **jääb** trükipoole probleemiks, sõltumata mootorist:
- tulemus on **lehe kaupa kas-või** ja kahe mootori vahel pöördub mõlemat pidi
  (`1642_16_..._0004`: llama.cpp +24, teenus 0). Noatera ei ole mootori oma,
  see on mudeli oma.
- `<cs>` katvus treeningandmetes on õhuke (punkt 3, mõõdetud).
- teenuse eeltöötlus on treeningust nihkes (punkt 1).
- mõõtmisalust ei ole üldse (punkt 4).

---

## 0b. Sihi seadmine: `<m>` on ainus sine qua non

**Kasutaja otsus (27.08):** marginaalia on ainus märgend, mis peab tingimata
olema. Kõik muu on debateeritav.

See ei ole pisidetail — see lahendab kolm lahtist küsimust korraga ja muudab
järjekorda.

### Mida see kohe otsustab

| märgend | otsus | miks |
|---|---|---|
| `<m>` | **puutumatu, kõik alluvad sellele** | siht |
| `<i>` **`<m>` sees** | **eemalda** | vasturääkiv (punkt 3e), ja vasturääkivus istub täpselt sihtmärgendi sees |
| `<cs>` | eemalda või edasi lükata | hajus märgendus (punkt 3c), nõuab otsustust |
| `<i>` põhitekstis | esialgu jätta | järjekindel, 94 teost; suur hoob, mida hoida varuks |
| `<pb/>` | jätta | struktuurne, üheselt mõistetav, 62 % lehtedest on kaksikpoognad |
| `<fn>`, `<noodid>` | jätta | **funktsionaalsed** – hoiavad mudelit rajal, `<noodid>` väldib loopi |

**`<i>` eemaldamine `<m>` seest muutub A/B-katsest lihtsalt õigeks teoks.**
Punkti 3e loogika oli: mehaaniliselt saab normaliseerida ainult ühes suunas
(ära võtta saab, juurde panna ei saa). Kui marginaalide kursiiv ei ole
kohustuslik, siis see suund ongi õige suund — vasturääkivus kaob, siht jääb.

**✅ TEHTUD 27.08.** `convert_marginalia.strip_italics_in_marginalia()`,
kutsutud `clean_markup`-i püsipunkti-tsüklist. Mõju kogu korpusele mõõdetud:

| | enne | pärast |
|---|---|---|
| `<i>` `<m>` sees | **4 986** | **0** |
| `<i>` väljaspool `<m>` | 9 812 | 9 812 (muutumatu) |
| `<m>` tage kokku | 8 505 | 8 505 (muutumatu) |
| muutunud lehti | — | 356 / 1113 |

Tähelepanuväärne: **34 % kõigist `<i>` tagidest oli marginaalide sees**. Ehk
varasem „`<i>` on 14 798 tagi, hästi kaetud" oli kolmandiku võrra
üles puhutud marginaalide kursiivist — märgendusest, mis oli ise vasturääkiv.
Tegelik põhiteksti `<i>` katvus on 9 812.

### Mida see ütleb mahuküsimuse kohta

Punkt 6 küsib, kas 51 M parameetrit on liiga vähe, sest nad kannavad
transkriptsiooni *pluss* märgendust. **Lihtsustamine on odavam kui maht.**
`<cs>` ja `<m>`-sisese `<i>` väljaviskamine vähendab õpitavat otsustuskoormust
ilma ühegi GPU-tunnita — ja teeb seda enne, kui r=64 üldse proovitakse.

**Järjekord peab seega olema: kõigepealt märgendite lihtsustamine, siis rank.**
Vastupidises järjekorras mõõdaks r=64 katse osaliselt seda, kui hästi suurem
adapter suudab vasturääkivat signaali pähe õppida — mis ei ole see, mida
teada tahame.

### Mida see ütleb mõõtmise kohta

Punkti 4 esimene kiht (10 GT-lehte) oli tasakaalustatud: 4 marginaalirohket,
2 ilma, 2 kursiivi/`<cs>`, 1 tabel, 1 hõre. **See tasakaal ei vasta enam
sihile.** Kui `<m>` on ainus kohustuslik, siis:

- **põhimõõdik on marginaalide recall ja marginaalide sisu CER**, mitte lehe
  koond-CER;
- holdout peab olema marginaalirohke — nt 6 lehte ≥20 `<m>`, 2 keskmist,
  2 ilma (kontrollgrupp valehäirete jaoks);
- `<cs>`-holdout kaob mõttetuks, kui `<cs>` treeningust välja läheb.

Ja Menii tõuseb: see on korpuse marginaalirikkaim ja raskeim materjal, ehk
täpselt see, mille peal siht seisab või kukub.

---

## 0c. Strateegia: kaks mudelit, mõlemad otse baasi pealt

**Kasutaja otsus (27.08):** kõigepealt suur uus Kurrendi treening, siis uus
trükimudel — **otse baasi peale**, mitte 1. etapi adapteri jätkuna. Sihiseis:

| mudel | baas | adapter |
|---|---|---|
| **käsikiri** | `unsloth/Qwen3.5-9B` | r=64 Kurrent |
| **trükk** | `unsloth/Qwen3.5-9B` | r=64 trükk + markup |

### Miks see on õige

1. **Kaotab kaheastmelise ahela probleemi juurtega.** Punkt 6 kirjeldab, kuidas
   praegune r=16 adapter kannab korraga 1. etapi transkriptsiooni ja 2. etapi
   märgendust. Baasilt treenimine teeb selle küsimuse olematuks — ei ole vaja
   merge'i, ei ole vaja `--lora-rank` lippu vana adapteri peale, ei ole vaja
   arutada, kumb pool 51 M parameetrit ära sööb.
2. **Sümmeetria.** Mõlemat mudelit saab edaspidi ühtemoodi uuendada, mõlema
   retsept on sama kuju, ja kogemus kandub ühelt teisele.
3. **Operatiivne boonus, mida tasub kohe kaaluda.** Kaks adapterit **sama
   baasi peal** tähendab, et teenus võib hoida mälus ühte baasmudelit ja
   vahetada adapterit. Praegu hoiavad kaks eraldi teenust 25,0/32,6 GB.
   See ei ole selle plaani osa, aga baasilt treenimine teeb selle võimalikuks;
   kaheastmeline ahel ei teeks.

### Aga see toob tagasi ühe kadunud sõltuvuse: **kreeka keel**

Baasilt treenimine tähendab, et 1. etapi 1500 lehte ei ole enam kaudselt
kaasas — nad tuleb treeningkomplekti **otseselt panna**, või nende sisu kaob.
Ja seal on üks asi, mida markup-korpus ei asenda.

**Mõõdetud 27.08:**

| | etapp 1 (`data/lehekyljed`, 1500 lk) | markup (`data/vutt`, 1113 lk) |
|---|---|---|
| kreeka tähemärke | **225 862** (11,9 %) | 13 734 (0,6 %) |
| lehti, kus kreekat leidub | 671 (45 %) | 297 (27 %) |
| **lehti, kus üle 20 % kreekat** | **559** | **6** |

Ja kogu see kreeka toetub **ühele teosele**: Gezeliuse leksikon on
**720 lehte ehk 48 % 1. etapi andmestikust**, neist 549 kreekarikast.
**Kreekarikkaid lehti väljaspool Gezeliust: 0.**

> **Ehk: kui uus trükimudel treenitakse ainult `data/vutt` peal, kaob
> 94 % kreeka treeningsignaalist ja jääb kuus lehte.** Kreeka on kõige
> suurem auk ja praegu katab seda üks märgendamata teos.

### Ja siis tuleb Gezeliust segada — see ei ole tasuta

Gezelius on **märgendamata**: puhas transkriptsioon, ilma `<m>`, `<i>`,
`<pb/>`-ta. Kaks konkreetset konflikti:

**(1) Märgendamata ≠ märgendivaba.** Leksikoni leht *tõenäoliselt* ei sisalda
marginaale (see on kaheveeruline sõnastik), ehk „ei ühtegi `<m>`" võib olla
päris õige silt. Aga ladina glossid (`radius rotae`, `tibiale, ocrea`) on
trükis peaaegu kindlasti kursiivis kreeka märksõnade kõrval, ja need on
märkimata. Ehk **`<i>` osas on Gezelius sama vasturääkiv signaal**, mille me
just `<m>` seest välja võtsime — 720 lehe jagu. Enne segamist tuleb otsustada:
kas Gezelius saab `<i>`-märgenduse, või jäetakse ta `<i>`-vabaks teadlikult
(ja siis on `<i>` katvus veelgi ebaühtlasem).

**(2) Poolitusmärk on kahes korpuses erinev — ja see on suurem probleem, kui
paistab.** Juhis (`prompt.py`) tunneb kahte: `-` (Antiqua) ja `⸗` (Fraktur).
Tegelikkus:

| poolitusmärk rea lõpus | etapp 1 | markup |
|---|---|---|
| `¬` U+00AC | **8 517** | **3 961** |
| `-` ASCII | 149 | 12 514 |
| `⸗` U+2E17 | 1 143 | 315 |

`¬` (loogika eituse märk!) esineb **12 478 korda ja ei ole juhises üldse
dokumenteeritud**. VUTT-i sees on ta koondunud: **12 teost 103-st**, ja need
12 kasutavad teda peaaegu eranditult (`1706-wilde-templa`: `¬` 622 vs `-/⸗` 5).
Ehk sama muster mis marginaalide kursiivil — **teose sees järjekindel, teoste
vahel vastuolus**, tõenäoliselt transkribeerija harjumus.

NB: need 12 `¬`-teost kattuvad suuresti kreekarikaste filosoofiadisputatsioonidega
(`1690-liber-philosophus`, `1696-9`, `1699-18`, `1693-7`) — ehk poolitusmärgi
valik ja kreeka materjal tulevad osalt samast käest.

**See on parandatav mehaaniliselt**, erinevalt kursiivist: `¬` → `-` või `⸗`
on ühene teisendus, kui otsustada kumb. Kirjatüübi järgi otsustamiseks on vaja
teada, kas leht on Fraktur või Antiqua; lihtsam reegel on **`¬` → `-`**, sest
`¬`-teosed on ladinakeelsed disputatsioonid (Antiqua). See tuleb teha
`clean_markup`-is, samamoodi nagu `<i>` eemaldus `<m>` seest.

### 0c-2. Gezelius on KAKS teost, mitte üks — ja ainult üks neist on probleem

Mõõdetud 27.08. `data/lehekyljed` 720 Gezeliuse lehte jagunevad:

| teos | lehti | read ainult kreeka | ainult ladina | **mõlemad samal real** |
|---|---|---|---|---|
| **Comenius-Ianua** | **273** | 45,8 % | 53,7 % | **0,2 %** |
| **Lexicon_exact** | **447** | 8,6 % | 9,5 % | **81,8 %** |

**Ianua on paralleelküljendus** — kreeka ja ladina eraldi ridadel, ladina osa
tavalises antiikvas. **Võib sisse võtta nagu on, märgendamata.** Kasutaja
hinnang leidis andmetest täpse kinnituse.

**Lexicon on põimitud** — 82 % ridadest sisaldab mõlemat kirja, sest struktuur
on „kreeka märksõna + ladina gloss". Just siin tekkis pseudomärgenduse mõte.

### 0c-3. Pseudomärgendus Lexiconis — kas liiga julge?

**Ettepanek:** Lexiconis järgneb kreeka sõnale kursiivne antiikva, ehk
`0021`–`0440` vahemikus saaks `<i>` masinaga sisse panna.

**Reegel ise ei ole liiga julge.** 82 % vs 0,2 % segaridu tõestab, et kaks
teost on tüpograafiliselt päriselt erinevad, ehk eeldus, millel reegel seisab,
peab paika. Kolm asja tuleb aga enne ära teha või teadlikult otsustada.

**(1) Transkriptsioonis on homoglüüfid — ja need rikuvad täpselt selle
signaali, millel reegel töötab.**

Suurtähelised kreeka märksõnad on kirjutatud **ladina näoga tähtedega**:

```
KNΊΣΣA   →  K, N, A on ladina; ΊΣΣ on kreeka
KΌΓΧΗ, KOΔΡΆΝΤΗΣ, ΚOIΛΊΑ, ὈMΦAΛῸΣ, ὌNAP
```

| | kreekat sisaldavaid sõnu | **segatud (ladina täht kreeka sõnas)** |
|---|---|---|
| **Lexicon** | 26 650 | **1 472 (5,5 %)** |
| **Ianua** | 19 888 | 8 (0,0 %) |

Sagedasemad: `A` 374, `N` 361, `P` 324, `O` 286, `M` 268, `K` 218 — täpselt
need, kus kreeka ja ladina suurtäht on identsed (Α/A, Ν/N, Ρ/P, Ο/O, Μ/M,
Κ/K). Ja need on **märksõnades**, ehk lehe kõige tähtsamates tokenites.

**See on iseseisev viga, sõltumata pseudomärgendusest:** mudelile õpetatakse
kreeka koha peal ladina koodipunkte. Parandus on mehaaniline ja ühene —
*ladina täht kreekat sisaldava sõna sees → kreeka vaste*. Prototüüp töötab:

```
enne:   <i>KN</i>ΊΣΣ<i>A vel</i> κνῖσα, ἡ, <i>nidor, dicitur etiam de</i>
pärast: ΚΝΊΣΣΑ <i>vel</i> κνῖσα, ἡ, <i>nidor, dicitur etiam de</i>
```

**Ianua on puhas** (8 juhtu, needki rooma numbrid `ΧΧΧV`) — parandust ei vaja.

**(2) Maht: pseudomärgendus muudaks `<i>` tähenduse korpuses ära.**

| | `<i>` spanne | lehti | lehe kohta |
|---|---|---|---|
| VUTT põhitekst (pärast `<m>`-puhastust) | 9 812 | 1113 | 9 |
| **Lexicon 0021–0440 pseudomärgendatuna** | **12 773** | 420 | **30** |

Ehk Lexicon annaks **57 % kõigist `<i>` tagidest ühendkorpuses**. Ja tähendus
ei ole sama: VUTT-is on `<i>` rõhutus, tsitaat või pealkiri jooksvas tekstis;
Lexiconis on ta sõnastiku ladina glossiveerg. **Sama märgend, teine funktsioon,
ja sõnastik võidaks 57:43.**

See on täpselt see haigus, mille me just `<m>` seest välja lõikasime
(4 986 vasturääkivat `<i>`) — tagasi toodud 2,6× suuremas mahus.

**(3) Jääkvead pärast homoglüüfiparandust** (näha prototüübis):
- päise rida `<i>KN</i> ΚΟ 193` saab ikka spani — esimene rida tuleb välja jätta;
- `<i>vox ranae. (ex sono dict</i>.)` — sulg jääb sisse, punkt välja, ehk span
  ei kattu trükitud kursiiviga täpselt;
- poolitus lõhub ühe trükitud kursiivijooksu kaheks spaniks (`spe¬` / `ciem`).

### 0c-4. Tehtud 27.08: eksport ja skript

**Lexicon EI OLE VUTT-is.** Kontrollitud varukoopiast: seal on
`1648-23 J.A. Comenij Janua` (184 lk, „Toores") ja
`1690-gezelius-posselius-familiarum-colloquiorum` — Lexiconit ei ole.

`scripts/gezelius_lexicon.py` teeb kolm asja, hoides need **rangelt lahus**:

| käsk | mis | mis EI ole |
|---|---|---|
| `--export DIR` | 447 lehte VUTT-i kujul (jpg + txt + json + `_metadata.json`), tekst **homoglüüfiparandusega** | ei sisalda pseudomärgendust |
| `--pseudo DIR` | sama tekst + `<i>` reegel, lk 21–440 | **sünteetiline GT**, ei lähe VUTT-i |
| `--valideeri KÄSITSI PSEUDO` | recall/täpsus käsitsi märgendatud lehtede vastu | — |

**Valmis kataloogid:** `data/export/gezelius-lexicon/` (124 MB, 447 lehte,
8 tühja) ja `data/export/gezelius-lexicon-pseudo/` (12 464 `<i>` spani).
Valideerimislehed: `data/export/valideerimis-lehed.txt` (10 juhuslikku,
seed 20260827).

**Homoglüüfiparandus tuli kahes jaos.** Põhireegel (ladina täht kreekat
sisaldava sõna sees) parandas 1 472 sõna, aga ei näinud sõnu, mis on
**tervenisti** ladina homoglüüfidest — neid oli 198, neist 185 lehepäises
(`OP` → `ΟΡ`). Lisatud eraldi reegel, mis puudutab ainult päise
**kahetähelisi** markereid; `INDEX.` lk 0440 jääb õigesti puutumata.

**Ja reegel andis kohe ühe õppetunni sünteetilise GT kohta.** Esimene versioon
märgendas 39 üksikut suurtähte lehe jalal (`<i>A</i>`, `<i>B</i>`, `<i>C</i>`,
`<i>D</i>`, iga 16 lehe järel) — need on **poogna signatuurid**, trükis
püstkirjas. Kreekas ei ole C-d ega D-d, ehk vea sai tagantjärele ära tunda;
aga ilma selleta oleks 39 vale spani läinud treeningusse iseendaga täiesti
kooskõlas. Täpselt see, mille eest allpool hoiatatakse. Parandatud, jäi 0.

**Piirang, mida ei saa siin parandada:** ekspordi pildid on **958×1654
(1,6 Mpx)**, VUTT-i tavapildid on ~3200×2500 (8 Mpx) — **5× vähem piksleid**.
Originaale `data/raw`-is ei ole. Kreeka diakriitika (rõhud, hengused) on
täpselt see, mis madalal resolutsioonil kaob. **Kui kõrgema eraldusvõimega
skaneering on kuskil olemas, tasub VUTT-i tõsta see, mitte need.**

### Soovitus

| mis | otsus |
|---|---|
| **Ianua 273 lk** | **võta sisse kohe, märgendamata** — andmed kinnitavad |
| **Homoglüüfide parandus Lexiconis** | **tee ära igal juhul** — transkriptsiooniviga, mitte märgendusküsimus |
| **Pseudomärgendus** | **kõik 420 lehte või mitte ühtegi** — pool oleks sama haigus mis 14-vs-30 marginaalide kursiiv |

**Üks hoiatus, mis on selle idee juures spetsiifiline.** Pseudomärgendus on
**sünteetiline ground truth**. Kui reegel on peenelt vale, on viga
*süstemaatiline ja iseendaga kooskõlas* — ta ei paista lossis, ei paista
valideerimisel, ei paista silmaga andmeid sirvides. Ta paistab **puhta
andmestikuna**. Seepärast: **märgenda 10 Lexiconi lehte käsitsi ja võrdle
reegli väljundiga**, enne kui 420 lehte sisse läheb. Tunni töö, ja ainus asi,
mis sünteetilise GT puhul aitab.

**Ja kuna `<m>` on ainus sine qua non** (punkt 0b): kui pseudomärgendus läheb
sisse, tuleb mõõta, kas `<i>` VUTT-tüüpi lehtedel halveneb. Kui halveneb, on
vastus juba plaanis — `<i>` eemaldamine kogu trükikorpusest oli niikuinii
varuhoob.

### Kreeka ja `<cs>` — üks asi, mis läks õigeks

Kontrollitud: **kreeka tähemärke `<cs>` sees on 0** (kogu 13 734-st). Ehk
kreeka-ladina koodivahetust ei ole kunagi `<cs>`-ga märgitud — see on
järjekindel väljajätt, mitte hajus märgendus. `<cs>` eemaldamine (punkt 3c)
ei puuduta kreekat.

### Mida see nõuab enne treeningut

| # | mis | miks |
|---|---|---|
| a | Ianua (273 lk) sisse märgendamata; Lexiconi (447 lk) `<i>`-otsus eraldi | vt 0c-2, 0c-3 |
| a2 | **Homoglüüfide parandus Lexiconis** (1 472 sõna, 5,5 %) | transkriptsiooniviga, sõltumatu märgendusest |
| b | `¬` normaliseerimine `clean_markup`-is | 12 478 esinemist, juhises puudub |
| c | Treeningkomplekt = `data/vutt` + `data/lehekyljed` (~2 613 lk) | ilma selleta kaob kreeka |
| d | Sama juhis (`INSTRUCTION`) mõlemal — see on juba nii | `train.py` kasutab sama `INSTRUCTION`, ehk 1. etapp treeniti markup-juhisega ilma markupita. Vasturääkivus on **juba praeguses mudelis sees** |

**Punkt (d) väärib tähelepanu:** `train.py:33` impordib sama `INSTRUCTION`-i
mis `train_markup.py`. Ehk juba 1. etapis öeldi mudelile „transkribeeri VUTT
XML märgendusega" ja anti vastuseks märgenditeta tekst — 1500 lehe jagu.
See ei ole uus risk, mille baasilt-treenimine tekitab; see on olemasolev risk,
mille baasilt-treenimine **nähtavaks teeb ja parandada laseb**.

### Ajakulu

Mõõdetud `checkpoints-markup-20260722` pealt: **27,8 s/samm**. Komplekt
~2 613 lk, `bs=1 × grad_acc=8`, 2 epohhi → ~653 sammu → **~5 h**.
Ehk baasilt treenimine ei ole kallis jooks; Kurrendi 12,3 h/epohh on kallis
jooks. Trükipoolel mahub A/B (nt r=32 vs r=64) ühte ööpäeva.

---

## 1. Teenuse eeltöötlus treeninguga kokku — esimene samm

`kataloogi-jalgimine-ja-ocr.py` annab protsessorile toorpildi, mis skaleerib
**BICUBIC**-uga. Treeningandmed valmistati **LANCZOS**-iga
(`build_vutt_dataset.py` → `prepare_image`). Teenus on ise treeningust nihkes.

Lahendus: kutsu `imaging.fit_to_grid()` enne protsessorit.

**Ootus, ettevaatlikult sõnastatud:** teadaolevalt parandab marginaalide
recall'i. v1 lubas „tasuta ~7 %" — see number tuleb **8 lehelt** (`<m>` 200 vs
216) ja on liiga kitsas alus põhjuslikuks väiteks. Enne kasutuselevõttu jooksuta
vana vs uus eeltöötlus **külmutatud 143 lehe peal** (punkt 4) ja vaata tabelit.
See ei maksa treeningaega ega inimtundi.

---

## 2. Pildieelarve — langetatud sondiks

Marginaalirida on treeningresolutsioonis (2560×1952) ~43 px kõrge, x-kõrgus
~15 px; üks visuaaltoken katab 32×32 px, ehk rida on **1,36 tokenit kõrge**.
Varu ei ole. Aga varu puudumine ei ole tõestatud põhjus:

- **lõikekatse on juba tehtud, mõlemas variandis.** Sama veerg samas
  pikslitiheduses eraldi pildina: 24 marginaali. Originaaltiheduses: 25.
  Terve lehe osana: 0. Vastus „kas mudel oskab neid glüüfe lugeda" on
  **jah**, ja see vastus on olemas — uut katset ei ole vaja.
- ja see kadu oli **llama.cpp oma** (punkt 0). Transformersi ahel luges
  sama lehe ääreveeru terve lehe pealt välja.

**v1 otsustuspuu ja järeldus „tükelda leht" on siit kustutatud.** Lõikekatse
tõestab, et regionaalne crop on hea *diagnostiline sond*. Ta ei tõesta, et
tootmisahel peaks lehti tükeldama — see tooks kaasa lugemisjärjekorra,
kattuvad lõigud, duplikaadid, `<m>`/`<i>`/`<cs>` struktuuri taastamise ja
küsimuse, kuidas üldse automaatselt teada, et täisleht jättis midagi vahele.
Kui teine pass kunagi tuleb, on see **eraldi tootmisarhitektuuri otsus**,
mitte selle plaani järeldus.

### Mis eelarvest alles jääb

**Kõva takistus püsib.** `train_markup.py:212` –
`UnslothVisionDataCollator(..., max_seq_length=8192)`. Kõigi 1113 treeninglehe
peal mõõdetud:

| | tokeneid |
|---|---|
| mediaan | 6893 |
| 90. protsentiil | 7442 |
| **max** | **8087** |
| max_seq_length | **8192** |

**Praegu ei kärbita ühtegi lehte — varu on 105 tokenit.** See on omaette risk
sõltumata marginaalidest: veidi pikem transkriptsioon lõikab lehe lõpu vaikselt ära.

| eelarve | visuaaltokeneid | lehti üle 8192 | vajalik max_seq_length |
|---|---|---|---|
| 5,12 Mpx (praegu) | ~5000 | 0 | 8192 |
| 6,0 Mpx | ~5860 | **236 (21 %)** | ≥ 8947 |
| 8,0 Mpx | ~7810 | **1113 (100 %)** | ≥ 10897 |

`max_seq_length` 8192 valiti algselt OOM-i tõttu. Eelarve tõstmine on
**mälu-, mitte ajaküsimus** ja võib RTX 5090 peal üldse mitte mahtuda.
Eelarve on ka andmestikku sisse küpsetatud (`prepare_image` skaleerib kettale),
seega muutmine tähendab andmestiku uuesti ehitamist.

### Katse B, ümber defineeritud

Aja 8 lehte transformersi ahelaga 8 Mpx juures. **Küsimus ei ole enam „kas
glüüfid muutuvad loetavaks"** — see on vastatud. Küsimus on kitsalt: **kas
suurem täislehe tokenitihedus muudab seda tähelepanu/küljenduse noateraefekti?**

- *ei muuda* → resolutsiooniteema saab lõplikult kõrvale panna
- *muudab* → siis alles tasub mälupiiriga maadelda

~20 min GPU-d, treenida pole vaja. Prioriteet: madal, mitte esimene.

---

## 3. `<cs>` katvus — mõõdetud 27.08 õhtul

`<cs>` (koodivahetus Fraktur↔Antiqua) oli v1-s ainult üks rida holdout'i
valikureeglis. See oli liiga vähe. Sama statistika mis `<m>` puhul, üle
`data/vutt/metadata.csv` (1113 lk):

| tag | kokku | lehti kus esineb | teoseid |
|---|---|---|---|
| `<i>` | 14 798 | 921 (83 %) | — |
| `<m>` | 8 505 | 527 (47 %) | 65 |
| **`<cs>`** | **821** | **193 (17 %)** | 78 |

Tiheduse jaotus: ≥3 `<cs>` lehel — 97 lehte; ≥5 — 61; ≥10 — **23**; ≥20 — **5**.
Kontsentreeritud: `1693-7 De origine Livonorum` 45, `1644 becker-linteum` 37
(8 lehel!), kaks *Moscoviae historia* varianti 36+36.

**Järeldus:** `<cs>` on `<i>`-st **18× haruldasem** ja tihedaid näiteid on
kahekümne kandis. See kinnitab varasemat hinnangut
(`markup_vs_transcription_quality.md`: `<i>` ei ole andmenappus, `<cs>` on) —
nüüd numbriga. `<cs>` on **päris andmekatvuse auk**, tõenäoliselt suurem kui Menii.

**NB, mida see EI ole:** juurdluse „`<cs>` jääb poole peale" (131 → 57) oli
**llama.cpp vs teenus**, ja trükipool jääb transformersile. See konkreetne
regressioon on olematu. Andmenappus on sellest sõltumatu ja päris.

**Tööjärg:** kui `<cs>`-rikast materjali märgendada, siis eelistatult
Fraktur-põhitekstiga teosed, kus ladina tsitaadid on sees — sama muster mis
`becker-linteum`.

### 3b. Kogu märgendi-inventuur, teoste kaupa (27.08 õhtul)

Lehtede arv ei ole ainus mõõt — **mitmest teosest** märgend tuleb, otsustab,
kas mudel saab üldistada või jätab pähe:

| tag | kokku | lehti | % lk | **teoseid** |
|---|---|---|---|---|
| `<i>` | 14 798 | 921 | 83 % | 94 |
| `<pb/>` | 694 | 694 | 62 % | 75 |
| `<m>` | 8 505 | 527 | 47 % | 66 |
| `<cs>` | 821 | 193 | 17 % | 78 |
| `<fn>` | 114 | 62 | 6 % | **3** |
| `<b>` | 44 | 17 | 2 % | **2** |
| `<noodid>` | 12 | 10 | 1 % | **1** |

**Punkti 3 sõnastust tuleb parandada:** `<cs>` on kõige suurem auk *nende
märgendite seas, mis esinevad paljudes teostes* (78!). `<fn>`, `<b>` ja
`<noodid>` tulevad 1–3 teosest, ehk mudel ei õpi neist reeglit, vaid jätab
konkreetsed leheküljed pähe.

**AGA vähene katvus ei tee neist filtreerimiskandidaate.** `<noodid>` on
**funktsionaalne, mitte stiililine**: ilma selleta satub mudel noodikirja peal
segadusse ja **läheb loopi**. Need ~10 lehte peavadki treeningus olema, nende
ainus ülesanne on õpetada „siin on noodid, pane marker ja liigu edasi".
Sama loogika kehtib osaliselt `<fn>` kohta. Filtreerimise arutelu käib ainult
nende märgendite kohta, mis nõuavad **otsustust** (`<cs>`, osalt `<i>`), mitte
nende kohta, mis hoiavad mudelit rajal.

### 3c. Kas `<cs>` üldse treeningust välja filtreerida?

Ettepanek: filtreerida `<cs>` välja ja jätta `<pb/>`, `<m>`, `<i>` — need on
visuaalselt otsustatavad, `<cs>` nõuab otsust „see on koodivahetus".

**Mis seda toetab — järjekindluse mõõtmine.** Teostest, kus `<cs>` üldse esineb
ja mis on ≥5 lehekülge (70 teost), on **67-l `<cs>` vähem kui pooltel
lehtedel**:

| teos | `<cs>`-lehti / kokku |
|---|---|
| `1690-liber-philosophus` | **2 / 50** |
| `1696-30 Exercitatio politica de majestate` | **4 / 47** |
| `1647-1 Oratio panegyrica` | 5 / 43 |
| `1707-17 Dissertatio philosophica` | 4 / 24 |

Osa sellest on õiguspärane (puhtladina Antiqua-teoses ei olegi koodivahetust).
Aga **2/50 ei ole „teoses on kaks ladina kohta"** — see on hajus märgendus.
Ehk mudelile antakse **vasturääkivat treeningsignaali**: sama visuaalne muster
on kord tagitud, kord mitte. See on halvem kui andmenappus.

**Mis selle vastu räägib — kaks asja, mis ei ole ilmsed.**

1. **`<cs>` ja `<i>` on sama visuaalne kanal.** Mõlemad on kirjatüübi
   eristused. Kui õpetame mudelit *ignoreerima* Antiqua-vs-Fraktur vahet
   (jämedam eristus) ja samal ajal *tähele panema* kaldkirja (peenem eristus
   samas kanalis), võib see `<i>`-le tagasi lüüa. See ei ole tõestatud,
   aga see on realistlik ja seda saab mõõta.
2. **Koodis pöörduv, andmetes mitte.** `clean_markup` puudutab ainult
   treeningteksti — VUTT-i salvestatud tekst jääb puutumata, ja eemaldamine on
   üks rida `UNWRAP_TAGS`-is. **Aga:** kui mudel lakkab `<cs>`-i tootmast, ei
   näe inimene seda enam mustandis ega lisa seda enam käsitsi → iga uus
   „Valmis" leht tuleb ilma. Korpus kaotab katvuse **jäädavalt**, ja tagasitee
   maksab käsitsi ümbermärgendust.

   Ainus, mis pöörduvuse alles hoiab, on **VUTT-i märgendusjuhis** — kui seal
   jääb `<cs>` nõudeks sõltumata sellest, mida mudel toodab, siis on otsus
   igal ajal tagasi keeratav.

**Soovitus: tee seda katsena, mitte poliitikamuudatusena.** Trükijooks on
mõõdetult ~2 h (punkt 6), ehk sama fixture, mis `r16` vs `r64`:

| | baas | andmed |
|---|---|---|
| **A** | merged stage-1 | markup `<cs>`-iga (praegu) |
| **B** | merged stage-1 | markup ilma `<cs>`-ita |

Mõõda punkti 4 alusel **eraldi**: (a) `<i>` täpsus — kas ignoreerimise õpetamine
lõi tagasi; (b) transkriptsiooni CER — kas otsustamiskoormuse äravõtmine
parandas põhiteksti. Kui B võidab mõlemal, on otsus tehtud tõendi, mitte
tunde põhjal. **Eeldus on jälle punkt 4** — ilma holdout'ita ei ole vahet näha.

**Kolmas tee, mida tasub kaaluda:** eemalda `<cs>` mitte üldse, vaid **nendest
teostest, kus märgendus on hajus** (nt < 30 % lehtedest tagitud). See võtab ära
vasturääkivuse, aga jätab alles need ~10 teost, kus märgendus on järjekindel, ja
säilitab mudeli võime. Kallim implementeerida, aga ei ohverda katvust.

### 3e. `<i>` ei ole nii süsteemne kui paistis — mõõdetud 27.08

Kasutaja tähelepanek: `<i>` on `<cs>`-ist palju süsteemsem, aga ka seal on
ebajärjekindlust — **mida teha, kui terve tekst on kursiivis, või kui KÕIK
ääremärkused on süsteemselt kursiivis?** Sellistel juhtudel jäetakse sageli
märgendamata. Mõõtsin. Tähelepanek peab paika, ja kõige valusamas kohas.

**Lehe tasandil probleemi EI ole.** 45 lehte on 90–100 % kursiiviga kaetud, ehk
täiskursiivsed lehed *on* märgendatud. Teoseid, kus on nii ~täiskursiivseid kui
0 %-lehti, on ainult 8, ja ainult üks teos (`1999-freilingshausen-2-tulpa`,
16 lk) on täiesti ilma `<i>`-ta.

**Marginaalide juures on probleem päris, ja see on teoste VAHEL.** Kas `<m>`
sisu on kursiivi pakitud (`<m><i>…</i></m>`), teoste kaupa (64 teost ≥15 `<m>`):

| `<m>` kursiivis | teoseid |
|---|---|
| 0–5 % (mitte ükski) | **22** |
| 5–30 % | 2 |
| 30–70 % | 3 |
| 70–95 % | **35** |
| 95–100 % (kõik) | 2 |

Teravalt **bimodaalne**: teos kas märgib kõik marginaalid kursiiviks või mitte
ühtegi. Teose *sees* on järjekindlus hea (64-st ainult 1 bimodaalne).

**Kas see on trükitüpograafia või märgendamispoliitika?** Bimodaalsus üksi ei
otsusta — ka päris trükikoja tava annaks sama kuju. Kaks tõendit ütlevad, et
vähemalt osa on märgendus:

**(1) Sama žanr, sama trükikoda, samad aastad, vastupidine kohtlemine.**
Academia Gustaviana oratsioonid 1633–1650: **14 teost „marginaalid ei ole
kursiivis", 30 teost „on"** — ja need on aastate kaupa läbisegi:

| aasta | EI kursiivis | ON kursiivis |
|---|---|---|
| 1636 | `1636-9 De castitate` | `1636-7 De bacchanalibus` |
| 1637 | `1637-1 Oratio enarrans` | `1637-2`, `1637-9` |
| 1638 | `1638-2`, `1638-21` | `1638-8`, `1638-17`, `1638-18` |

**(2) Viis teost, kus märgendus on tüpograafiliselt tagurpidi.** Põhitekst on
kursiivirikas, aga marginaalides ei ole ainsatki `<i>`-d:

| teos | põhitekst kursiivis | `<m>` kokku |
|---|---|---|
| `1636-9 Oratio de castitate` | **85 %** | 156 |
| `1646-7 In salutiferam nativitatem` | **85 %** | 103 |
| `1647-10 De sanctis angelis` | **70 %** | 133 |
| `1646-1 In laudem poeseos` | 28 % | 130 |
| `1700-2 Meletema academicum` | 25 % | 64 |

85 % kursiivse põhitekstiga teoses **antikva-marginaalid oleks tüpograafiliselt
tagurpidi** — marginaal seatakse kontrastse kirjaga. Need viis on kõige
tõenäolisemalt märgendusviga, kokku **~390 `<m>` tagi ehk ~5 % kõigist**.

**Miks see on tähtsam kui `<cs>`:** ebajärjekindlus tabab täpselt seda
märgendit, mille pärast kogu see plaan kirjutati. Mudel näeb tihedaid
ääreveerge, kus 30 teost ütlevad „paki iga rida `<i>`-sse" ja 14 ütlevad
„ära paki" — **ilma nähtava visuaalse vaheta**. Vasturääkiv signaal täpselt
kõige raskemas kohas. (Kas see seletab ka lehe-kaupa-kas-või käitumist, on
spekulatsioon — see sümptom puudutas marginaali *olemasolu*, mitte kursiivi.
Aga ebakindlus selles, kuidas marginaali vormistada, ei tee stabiilsust
paremaks.)

**Mida teha, järjekorras:**

1. **Vaata need 5 teost pildilt üle** (~30 min). Kui marginaalid on trükis
   kursiivis, on tegu 390 vale tagiga ja parandus on ühekordne.
2. **Otsusta juhis 14-vs-30 kohta.** See ei ole koodiküsimus: kas süsteemselt
   kursiivne ääreveerg märgendatakse või mitte? Vastus peab minema VUTT-i
   märgendusjuhisesse, muidu jaotus taastub.
3. **Mehaaniline normaliseerimine on võimalik ainult ühes suunas.** `<i>`-d
   saab `<m>` seest ära võtta (`clean_markup`-is, üks funktsioon), sest see ei
   nõua teadmist. Juurde panna ei saa, sest me ei tea tõde. Ehk kui otsus on
   „ära märgi", saab kogu korpuse kohe ühtlustada; kui „märgi", tuleb 22 teost
   käsitsi üle käia.
4. **Sama A/B fixture mis `<cs>`-il** (punkt 3c): treeni `<i>`-ga `<m>` sees ja
   ilma, mõõda marginaalide recall'i ja CER-i.

---

## 3d. Märgendite hügieen — leitud auk (parandatud 27.08)

**Küsimus oli: kas `<ann>` filtreeritakse ikka välja?** Jah —
`convert_marginalia.py:204` `UNWRAP_TAGS`, kutsutud `clean_markup` kaudu nii
`build_vutt_dataset.py`-st kui `train_markup.py`-st. Kontrollitud: praeguses
`data/vutt/metadata.csv`-s on **0 `ann`-märgendit**.

**Aga nimekiri oli lühem kui VUTT.** `UNWRAP_TAGS` kattis `ann1`–`ann4`;
VUTT-i varukoopias on ka **`ann5`–`ann14`** (30 esinemist, 3 failis). Need
oleksid lekkinud treeningandmetesse niipea, kui selline leht saab „Valmis".
**Parandatud:** `unwrap_tags` võtab nüüd lisaks `ann\d+` regexi.

**Suurem probleem püsib: tundmatu märgendi valvurit ei ole üldse.** VUTT-i
25 105 tekstifailis on märgendeid, mida `prompt.py` ei dokumenteeri ega ükski
filter ei koristata:

| tag | esinemisi | failides | mis see on |
|---|---|---|---|
| `fb` | 841 | **1** | `<fb>1</fb>` – Gezeliuse colloquia |
| `sup` | 282 | 31 | ülakiri, `<sup>mus</sup>`, `<sup>1)</sup>` |
| `sub` | 14 | 2 | alakiri |
| `u` | 6 | 1 | allajoonitud |
| `responsum`, `vero`, `Dominus` | 1 | 1 | ilmselgelt näpukad – `<` pandi ümber tavalise sõna |

Ükski neist ei ole praegu treeningandmetes, aga **miski ei takista neil sinna
sattumast** — piisab ühest „Valmis" märkimisest. Siis õpetame mudelile
märgendeid, mida juhises ei ole, ja väljund läheb VUTT-i tagasi.

**Vajalik parandus:** `build_vutt_dataset.py` peaks tundmatu märgendi peale
**valjult kaebama** (loendama ja loetlema, mitte vaikselt läbi laskma).
Tuntud inventar on täpselt see, mis `prompt.py`-s: `i, b, cs, m, fn, pb,
noodid`. Kõik muu on kas näpukas või uus juhisereegel — mõlemal juhul peab
inimene seda nägema.

---

## 4. Mõõtmisalus — kolm tasandit, üks neist külmutatud

Ilma selleta ei saa punkti 5 ega 6 tulemust tõestada; teeksime kalli jooksu ja
vaataksime tulemust silmaga, nagu täna.

### Kiht 1 — 10 lehte päris ground truth'iga (maksab 10 treeninglehte)

**Ankurdab absoluutset kvaliteeti** (CER, marginaalide täpsus). Ei anna täpset
korpuse-CER-i — kümme lehte selleks ei kõlba, ja seda ei tohi nii ka esitada.

Valikureegel: **tüüpilisi, mitte haruldasi**. Haruldase küljenduse väljajätmine
maksab treeningus rohkem. Jaotus: 4 marginaalirohket (≥20 `<m>`),
2 marginaalideta, 2 kaldkirja/`<cs>`-rohket, 1 tabel/register, 1 hõre leht.

### Kiht 2 — Menii challenge-set, ~5 lehte (kallis, aga vajalik)

Kui kõik Menii-tüüpi lehed lähevad treeningusse, ei ole pärast **millegagi
mõõta seda probleemi, mille pärast neid üldse märgendati**. Seega ~25 tehtud
lehest **20 treeningusse, 5 jäädavalt välja**. Need viis vastavad ainsana
küsimusele „kas uus treening parandas Menii-laadset küljendust".

### Kiht 3 — külmutatud 143 „Toores" lehte (maksab MITTE MIDAGI)

Ei ole niikuinii treeningus, sest inimene pole neid kinnitanud. Ei mõõda
absoluutset kvaliteeti — mõõdab **regressiooni**.

**Aga komplekt tuleb külmutada.** Praegu on see *päring*, mitte nimekiri:
`reocr_vutt.py` valib `status == "Toores"` + `updated_at >= --since` +
`type == print`. Iga jooksuga on see teine komplekt, ja mudel B võib mudelist A
parem paista lihtsalt seetõttu, et vahepeal lisandus 50 lihtsat lehte.

Külmuta `vutt-print-regression-v1`-na: lehe ID, pildi hash, praeguse
referentsväljundi hash **ja referentsteksti koopia** (VUTT-i tekst kirjutatakse
lehe uuel töötlemisel üle — päringu peale ei saa loota). Alus on olemas:
`data/vutt/reocr/markup-20260722-Q8_0/results.csv`, 143 rida, `leht` veerg.
Praegune koosseis: 59 lk Menii (1635-1), 15 lk 1632-1, 11 lk 1632-12,
9 lk 1638-39, ülejäänu sabas.

Uued „Toores" lehed võivad moodustada eraldi *rolling*-komplekti, aga
põhinumber tuleb alati samalt 143 lehelt.

### Kokkuvõttes

> **10 GT-lehte ankurdavad absoluutset kvaliteeti; ~5 Menii lehte mõõdavad
> sihtprobleemi; 143 külmutatud lehte tuvastavad regressioone.**

### Kolm koodiauku, mis blokeerivad kõik ülaltoodu

Neid v1-s ei olnud ja need on eeldus punktidele 1, 4, 5 ja 6:

| mis | seis |
|---|---|
| `reocr_vutt.py` | **ainult llama-server** (`--endpoint` kohustuslik, rida 142). Trükipoolel mõõtmiseks vajab transformersi backendi. |
| `make_holdout.py` | ainult Kurrent (`data/kurrent/metadata.csv`). Vajab VUTT-i varianti. |
| `train_markup.py` | ei oska holdout'i välja jätta üldse. |

---

## 5. Menii ja vaidluslehtede märgendamine — tööjärg, mitte ajakava

**Seis:** marginaalide kogus treeningus ei ole kõhn — 527 lehte 1113-st (47 %),
8505 `<m>`, 65 teosest. Kitsas on **tihedus ja žanr**:

| `<m>` tagi lehel | treeninglehti |
|---|---|
| ≥10 | 383 |
| ≥25 | 107 |
| ≥30 | **41** |
| ≥39 | **3** |

Menii lehed on 24–39 tagiga, ehk jaotuse 90.–100. protsentiilis, ja
**Frid. Menii ei ole treeningandmetes üldse** (0 lehte). Treening on valdavalt
Academia disputatsioonid ja oratsioonid, max 12 lk teose kohta; Menii on
ajalooraamat tiheda ääreveeruga — **tundmatu küljendus**, mitte lihtsalt
tundmatu teos.

**Argument ei ole enam „tootmismudel on Menii peal katki"** (ta ei ole, vt
punkt 0), vaid: mudel on selle küljenduse peal **noateral** — sama leht
pöördub mootorit vahetades mõlemat pidi. Noatera on treeningkatvuse tagajärg,
ja seda parandavad lehed, mitte pikslid.

**Piirang:** doktorandid töötavad oma materjalidega, seda järjekorda ei muuda.
Menii teeb kasutaja ise, kui tunde jagub. Ehk see punkt **ei ole ajastatav** —
aga prioriteedireegel on olemas:

> **Anna inimesele esimesena need lehed, kus kaks mudeliahelat lahku lähevad.**

Nimekiri: `docs/marginaalid-silmaga-vaadata.md` — 23 lehte, vaidlusalused read
kõrvuti. 143 re-OCR-lehest on **59 just Menii omad**. Seal on mudel ebakindel
ja sisu rikas. Aktiivõpe, mis ei maksa midagi peale inimtunni.

Neist ~25 tehtud lehest **5 jääb challenge-set'i** (punkt 4, kiht 2).

---

## 6. LoRA maht ja kaheastmeline ahel

### Asümmeetria on päris — aga mitte see, mis v1-s kirjas oli

| mudel | LoRA | treenitavaid | epohhe | näiteid nähtud | loss lõpus | lähtepunkt |
|---|---|---|---|---|---|---|
| **trükk** 20260722 | r=16, α=16 | **51,0 M** | 2 | ~2 144 | 0,036 | **eeltreenitud** (1500 lk transkriptsioon) |
| **Kurrent** 20260602 | r=64, α=64 | **203,9 M** | 2 | ~25 400 | 0,108 | puhas baas `unsloth/Qwen3.5-9B` |

**v1 väitis „12× väiksema sammuarvuga". See on tühi väide** — mõlemad on
2 epohhi, sama `grad_acc=8`, sama `bs=1`; 268 vs 3178 sammu tähendab ainult, et
andmestik on 11× väiksem. Sammuarv ei ole võrreldav mõõt; nähtud näidete arv
või epohhid on.

**Mis päriselt alles jääb, on kaks asja:**

**(1) Lossi ei saa otse võrrelda.** Trükk alustab soojalt: 1. etapi
transkriptsioonioskus on juba adapteris. Kurrent alustab külmalt. Trüki
loss 0,036 ei tõesta, et maht on piisav.

**(2) See r=16 adapter kannab KAHTE asja korraga.** `train_markup.py` laadib
`models/qwen3.5-ocr-lora-backup-20260527` ja treenib **sedasama adapterit
edasi** (`get_peft_model()` EI tohi järgneda). Ehk 51 M parameetrit hoiavad
nii 1. etapi transkriptsiooni kui märgendust; Kurrendi 204 M hoiavad ühte asja.
See on ainus tõsine mahuargument ja see seisab.

### Lahendus: liida 1. etapp baasi

```bash
venv/bin/python scripts/merge_lora.py models/qwen3.5-ocr-lora-backup-20260527
venv/bin/python scripts/train_markup.py \
    --base=models/merged/qwen3.5-ocr-lora-backup-20260527-bf16 --lora-rank=64
```

Siis kannab adapter ainult märgendust, transkriptsioon on kaaludes sees.
Arhitektuuriliselt palju puhtam kui vana adapteri lõputu edasiõpetamine.

**KOODIMUUDATUS ON VAJALIK.** `train_markup.py`-l ei ole `--lora-rank` lippu
ega `get_peft_model()` kutset üldse — ta OSKAB ainult olemasolevat adapterit
edasi treenida. `train_kurrent.py`-s on õige muster olemas:

| mida | `train_kurrent.py` |
|---|---|
| `LORA_RANK` konstant + `--lora-rank=` lipp | read 57, 65–66, 72 |
| tingimuslik `get_peft_model()` HF-baasi puhul | read 74, 129–139 |

Töö on väike (~20 rida), aga tingimus tuleb kirjutada nii, et olemasolev ahel
(checkpointist edasi) käitub täpselt nagu praegu.

### Katse peab olema kontrollitud — v1 oma ei olnud

v1 pakkus võrdlust *praegune continued r=16* vs *merged base + fresh r=64*.
Seal muutub **korraga kaks asja** (rank; ja edasitreenitud vana adapter →
värske adapter puhtal baasil). Kui r=64 võidab, ei tea, kumb selle põhjustas.

Õige katse:

| | baas | adapter |
|---|---|---|
| **A** | merged stage-1 BF16 | **värske r=16** |
| **B** | merged stage-1 BF16 | **värske r=64** |

Sama andmestik, sama split, sama seed, sama LR-graafik, sama efektiivne batch.
Alles siis saab öelda, kas rank aitas. `r=32` oleks hea keskpunkt, aga mitte
hädavajalik.

**Ja see on odav.** Mõõdetud `models/checkpoints-markup-20260722/` ajatemplitest:
checkpoint-5 kell 19:52, checkpoint-268 kell 21:54 → **~27,8 s/samm**, terve
jooks **~2 h**. Paar A/B jooksu on ~5 h GPU-d. Ei ole põhjust confounded katset
teha.

**Eeldus:** ilma punktita 4 ei ole tulemust millegagi mõõta.

---

## Kokkuvõttes, järjekorras

| # | samm | maksumus | blokeerib |
|---|---|---|---|
| 0 | **Suur Kurrendi treening** | ~25 h GPU | kõike muud |
| 1 | ✅ `<i>` välja `<m>` seest `clean_markup`-is (tehtud 27.08) | — | — |
| 2 | `¬` → `-` normaliseerimine `clean_markup`-is (punkt 0c/b) | ~1 h koodi | trükitreeningut |
| 3 | Tundmatu märgendi valvur `build_vutt_dataset.py`-sse (punkt 3d) | ~1 h koodi | andmete puhtust |
| 4 | Homoglüüfide parandus Lexiconis (punkt 0c-3) | ~1 h koodi | trükitreeningut |
| 4b | 10 Lexiconi lehte käsitsi → pseudomärgenduse reegli valideerimine | ~1 h inimtööd | pseudomärgenduse otsust |
| 5 | `reocr_vutt.py` transformersi backend + 143 lehe külmutamine | ~pool päeva koodi | 6, 8, 9 |
| 6 | BICUBIC → LANCZOS teenuses, A/B külmutatud 143 peal | ~1 h | mitte midagi |
| 7 | 5 kahtlase teose kontroll pildilt + juhiseotsus `<m><i>` kohta (punkt 3e) | ~30 min silmaga | VUTT-i märgendusjuhist |
| 8 | **Marginaalikeskne** 10-leheline GT-holdout + `make_holdout` VUTT-ile + `train_markup.py` holdout-tugi | 10 treeninglehte + kood | 9 |
| 9 | **Trükimudel otse baasilt**, `data/vutt` + `data/lehekyljed`, r=64 | ~5 h GPU | — |
| 10 | Menii ja vaidluslehtede märgendamine → järgmine ring | inimtunnid, ajastamata | — |

Edasi lükatud: `<cs>` eemaldamise A/B (punkt 3c), Katse B 8 Mpx sondina
(punkt 2), `<i>` eemaldamine ka põhitekstist, stage-1 merge (punkt 6 — muutub
tarbetuks, kui treenida otse baasilt).

Sammud 1, 2 ja 4 on odavad ja kohe. Samm 3 on **eeldus**, mida v1-s ei
märgatud: kogu mõõtmisaparaat räägib praegu ainult llama-serveriga, mida
trükipoolel ei kasutata.

**Juhtmõte:** iga samm on siin sellepärast, et ta teenib `<m>`-i. Mis seda ei
tee, on edasi lükatud.

---

## Mis v1-s valesti oli

| v1 väitis | tegelikult |
|---|---|
| „Katse A: lõika marginaaliveerg ja vaata, kas mudel loeb" | **juba tehtud, mõlemas tiheduses** (24 ja 25 marginaali). Vastus on olemas, katse kustutatud. |
| Otsustuspuu → „Tükelda leht" | Lõikekatse tõestab diagnostilist sondi, mitte tootmisarhitektuuri. Otsustuspuu kustutatud. |
| Marginaalide kadu õigustab pildieelarve/tükeldamise arutelu | Kadu oli **llama.cpp oma**; teenus luges `1635_1_0017` pealt 28 `<m>`, llama.cpp 0. Trükipool jääb transformersile. |
| „12× väiksem sammuarv" tõestab mahuasümmeetriat | Mõlemad 2 epohhi, sama batch; 268 vs 3178 = andmestiku suurus. Väide kustutatud. |
| r=16 vs r=64 võrdlus (continued vs fresh) | **Confounded** — kaks muutujat korraga. Õige on fresh r16 vs fresh r64 samalt merged baasilt. |
| „tasuta ~7 % marginaale" | Number tuleb 8 lehelt, liiga kitsas põhjuslikuks väiteks. Sõnastus pehmendatud, A/B 143 lehel enne kasutuselevõttu. |
| `<cs>` = üks rida holdout'i valikureeglis | **Mõõdetud: 821 tagi, 193 lehte (17 %), `<i>`-st 18× haruldasem.** Eraldi punkt 3. |
| „143 lehte" kui püsiv regressioonikomplekt | See on *päring*, mitte nimekiri — kasvab iseenesest ja rikub ajaloolise võrreldavuse. Tuleb külmutada. |
| „10 lehte ütlevad, kui hea mudel on" | 10 lehte **ankurdavad** absoluutset kvaliteeti; korpuse CER-i nad ei anna. |
| Menii lehed treeningusse | ~5 tuleb jäädavalt välja jätta, muidu pole sihtprobleemi millegagi mõõta. |
| — (puudus) | **`reocr_vutt.py` on llama-server-only, `make_holdout.py` Kurrent-only, `train_markup.py`-l pole holdout-tuge.** Kolm koodiauku enne igasugust mõõtmist. |
| „`<cs>` on suurim andmeauk" | Suurim auk **paljudes teostes esinevate** märgendite seas. `<fn>` (3 teost), `<b>` (2), `<noodid>` (1) on hullemad – need on päheõppimine, mitte reegel. |
| „`<i>` on 14 798 tagi, hästi kaetud" | **34 % neist (4 986) oli `<m>` sees** – vasturääkiv märgendus. Põhiteksti `<i>` on 9 812. |
| „Gezelius on üks märgendamata teos" | **Kaks teost.** Ianua (273 lk) on paralleelküljendus, 0,2 % segaridu — võib sisse võtta nagu on. Lexicon (447 lk) on põimitud, 81,8 % segaridu. |
| — (puudus) | **Lexiconi kreeka märksõnades on homoglüüfid**: 1 472 sõna (5,5 %) sisaldab ladina näoga tähti (Α/A, Ν/N, Ρ/P…). Ianua on puhas. Iseseisev transkriptsiooniviga. |
| — (puudus) | **Kreeka on suurim auk ja toetub ühele märgendamata teosele**: 559 kreekarikkast lehest 549 on Gezelius; väljaspool teda 0. Baasilt treenimine nõuab `data/lehekyljed` kaasamist. Vt 0c. |
| — (puudus) | **Poolitusmärk `¬` (U+00AC) esineb 12 478 korda ja ei ole juhises üldse.** VUTT-is 12 teoses 103-st. Mehaaniliselt parandatav. Vt 0c. |
| — (puudus) | `UNWRAP_TAGS` kattis ainult `ann1`–`ann4`, VUTT-is on `ann5`–`ann14`. **Parandatud.** Ja tundmatu märgendi valvurit ei ole üldse – `fb`, `sup`, `sub`, `u` ootavad lekkimist. |
| „`<fn>`/`<b>`/`<noodid>` on 1–3 teosest, ehk päheõppimine" | Tõsi, aga **`<noodid>` on funktsionaalne** – ilma selleta läheb mudel noodikirja peal loopi. Vähene katvus ei tee sellest filtreerimiskandidaati. |
| „`<i>` on korras, `<cs>` on probleem" | **`<i>` on `<m>` sees samamoodi vasturääkiv**: 22 teost ei märgi marginaale kursiiviks, 35 märgivad – sama trükikoda, samad aastad. 5 teoses on märgendus tüpograafiliselt tagurpidi (~390 tagi). Vt 3e. |
