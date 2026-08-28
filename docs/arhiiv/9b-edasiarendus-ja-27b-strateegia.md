# Qwen3.5 9B edasiarendus ja 27B strateegia

**Viimati uuendatud:** 2026-07-22  
**Staatus:** aruteludokument, mitte kinnitatud treeninguplaan

## Lähtekoht

Praegune Qwen3.5-9B trükimudel transkribeerib tähti juba hästi. Peamine
allesjäänud probleem on märgendamine ja mõne keeruka lehekülje käsitlemine
tervikuna: milline tekst on marginaal, kursiiv või koodivahetus, kuhu märgend
paigutada ning kuidas seostada põhitekst kõrvalelementidega. Mürastel lehtedel
esineb rohkem tähevigu, kuid see on ootuspärane ega ole praegu peamine
arendusküsimus.

Käsikirjamudeliga on juba olemas oluline empiiriline tulemus: LoRA rank 16
andis üsna kesiseid tulemusi, rank 64 aga selgelt paremaid. See ei tõesta, et
trükimudel vajab tingimata rank 64, kuid näitab, et adapteri maht võib selle
projekti keerukate visuaalsete oskuste puhul päriselt piirata.

Seetõttu ei ole veel põhjust järeldada, et 9B baasmudeli parameetrite arv on
pudelikael. Tõenäolisemad uurimiskohad on markup-andmestik, adapteri maht ja
õppeviis. Probleemi võimalikud põhjused on ka:

- pildi- ja tekstijärjendi ühine tokenieelarve;
- keerukate küljenduste vähene või ebaühtlane esindatus treeningandmetes;
- train/validation jaotuse ja raskusklasside põhise hindamise puudumine;
- treeningu hüperparameetrid ja LoRA konfiguratsioon;
- vision encoder'i kohandamise viis;
- kaheetapilise treeningu võimalik unustamine;
- prompt'i ja ground truth'i ebaselged või ebajärjekindlad küljendusreeglid.

27B mudel jääb perspektiivseks, kuid enne pilverenti tuleb selgitada, kui palju
on võimalik 9B mudelit praeguse riistvaraga parandada.

## Praeguse koodi tugevused

- Pildieelarve on tsentraliseeritud failis `scripts/imaging.py` ja seatud
  väärtusele 5 120 000 pikslit ehk ligikaudu 5000 visuaaltokenit.
- Treening kasutab 4-bitist baasmudelit, gradient checkpointing'ut ja LoRA-t,
  mis võimaldab 9B mudelit RTX 5090 32 GB kaardil treenida.
- Inferents kasutab BF16, mitte aeglasemat 4-bitist laadimist.
- Olemasolevast checkpoint'ist jätkamisel ei kutsuta `get_peft_model()` uuesti.
- Pildid laaditakse laisalt ja trüki- ning käsikirjaandmed on eraldatud.
- Trüki markup-andmestikus on praegu 1066 CSV-kirjet; käsikirjaandmestikus
  17 018 kirjet. CSV füüsiliste ridade arv on transkriptsioonide reavahetuste
  tõttu palju suurem ega näita lehekülgede arvu.

## Olulised piirangud praeguses treeningutorus

### 1. Pildi- ja väljundtokenite eelarve võib olla liiga kitsas

Treeningus on korraga:

- kuni umbes 5000 visuaaltokenit;
- pikk instruktsioon;
- kogu lehekülje transkriptsioon;
- `max_length=8192`.

See jätab pika lehekülje tekstile ligikaudu ainult 3000 tokenit või vähem.
Samas lubab testinferents kuni 4096 uut tokenit. `tokenizer.truncation=False`
ei tõesta iseenesest, et treeningnäide jõuab kadudeta kollatorist ja treenerist
läbi; `max_length=8192` võib ikkagi olla piirav.

See on esimene asi, mida tuleks mõõta. Vaja on skripti, mis arvutab kogu
andmestiku kohta:

- visuaaltokenite hinnangu;
- prompt'i tokenid;
- vastuse tokenid;
- kogupikkuse;
- 8192 piiri ületavate lehtede arvu ja failinimed.

Kui terviklehekülje vead koonduvad just pikkadele või suure resolutsiooniga
lehtedele, võib põhjus olla tokenieelarves, mitte 9B mudeli võimekuses.

### 2. Päris valideerimisjaotus puudub

Praegu läheb kogu CSV `train_dataset`-ina treenerisse. Treeninguaegset
`eval_dataset`-i pole ning `scripts/eval_model.py` kasutab väikest eraldi
GT-komplekti.

Jaotus peab olema teose, säiliku või autori põhine, mitte juhuslik lehekülgede
jaotus. Sama teose naaberlehed on visuaalselt liiga sarnased ja tekitaksid
andmelekke. VUTT-i loodud `metadata.csv` peaks selleks säilitama vähemalt
teose ID ja võimaluse korral keele, sajandi ning küljenduse tüübi.

### 3. Hindamine ei mõõda veel lehekülje struktuuri ega märgendamist

CER ja WER üksi ei kirjelda hästi terviklehekülje mõistmist. Kuna tähtede
transkriptsioon on juba hea, tuleb hindamisel lahutada vähemalt kaks tulemust:

- XML-märgenditest puhastatud teksti CER/WER;
- märgendite ja nende piiride täpsus.

Nii saab kindlaks teha, kas uus katse parandab markup'i ilma transkriptsiooni
halvendamata. Lisaks on vaja mõõta või vähemalt käsitsi märgendada:

- kas kõik tekstiplokid jõudsid väljundisse;
- lugemisjärjekorra õigsus;
- veergude järjekord;
- marginaalide asukoht ja `<m>` märgendus;
- `<i>`, `<b>`, `<cs>`, `<fn>`, `<pb/>` ja `<noodid>` täpsus;
- XML-i süntaktiline korrektsus;
- päiste, jaluste ja signatuuride käsitlus;
- lehekülje lõpu äralangemine või kordused.

Kasulik oleks moodustada 30–100 lehest „raskete lehtede” valideerimiskorpus,
kus iga leht kuulub ühte või mitmesse raskusklassi: kaks veergu, tihe
marginaalia, segakirjad, kreeka tekst, tabel, laiendus, kahjustatud skann jne.

### 4. Käsikirjamudeli üldtest kasutab vale prompt'i

`scripts/test_model.py` kasutab alati trükimudeli `INSTRUCTION` prompt'i ka
siis, kui anda `--dir data/test/hand`. Käsikirja testimiseks tuleb lisada
näiteks `--type print|hand`, mis valib vastavalt `INSTRUCTION` või
`KURRENT_INSTRUCTION`.

Enne mudelite või kvantide võrdlemist tuleb see parandada, muidu ei ole
käsikirja tulemused võrreldavad.

### 5. Checkpoint'ist jätkamine ja logimine pole täielikud

Checkpoint-kaustad on määratud, kuid treeningut ei käivitata kujul
`trainer.train(resume_from_checkpoint=...)`. Samuti on `report_to="none"`.
Järgmine versioon peaks salvestama vähemalt:

- kõik treeninguparameetrid JSON- või YAML-faili;
- train- ja validation-loss'i;
- VRAM-i maksimumi;
- näiteid ja tokeneid sekundis;
- checkpoint'id kindla sammude intervalliga;
- jätkamise argumendi `--resume`.

`scripts/train_pipeline.sh` vajab ka `trap`-i, mis käivitab kohaliku
`ocr-service` teenuse uuesti juhul, kui treening veaga katkeb.

## 9B mudeli parandamise prioriteetsed katsed

### Katse A: tokenipikkuste audit

Kõigepealt mõõta, mitte muuta mudelit. Tulemuseks peab olema CSV või raport,
mis seob iga lehe pildi-, prompt'i- ja vastusetokenite arvuga. Seejärel
võrrelda üle 8192 piiri minevaid lehti teadaolevate halbade tulemustega.

Võimalikud lahendused sõltuvalt tulemusest:

- tõsta treeningu `max_length` väärtust, kui VRAM lubab;
- vähendada prompt'i ilma reegleid kaotamata;
- vähendada pisut pildi tokenikulu lehtedel, kus tekst on jämedas kirjas;
- kasutada eri leheklassidele erinevat pildieelarvet;
- jätta väga pikad lehed eraldi pikema kontekstiga treeninguetappi.

Pildiresolutsiooni ei tohiks pimesi vähendada: see võib parandada
küljenduse tervikpilti, kuid halvendada väikeste tähtede, diakriitikute ja
marginaalide lugemist.

### Katse B: markup'i ja raskete küljenduste sihitud andmestik

Praeguse 1066-lehelise `data/vutt/metadata.csv` kiire loendus näitab, et
märgendite esindatus on väga ebaühtlane:

| Märgend | Avamärgendeid | Lehti, kus esineb |
|---|---:|---:|
| `<i>` | 13 597 | 880 |
| `<m>` | 6 847 | 487 |
| `<cs>` | 764 | 184 |
| `<pb/>` | 659 | 659 |
| `<fn>` | 114 | 62 |
| `<b>` | 44 | 17 |
| `<noodid>` | 12 | 10 |

`<pb/>` esialgne nulltulemus oli loendusskripti regulaaravaldise viga:
avamärgendite jaoks kirjutatud muster ei arvestanud nime järel olevat `/`
märki. Tegelikus `data/vutt/metadata.csv` failis on 659 `<pb/>` märgendit
659 eri lehel. `data/vutt-raw/` all on kokku 755 märgendit, millest 659 on
trükiste staatusega „Valmis” lehtedel ja jõuavad praeguse ehitusahelaga
andmestikku.

Kõigil 659 `<pb/>` näitel on täpselt üks leheküljepiir. Märgend paikneb
transkriptsioonis enamasti peaaegu keskel: mediaanne suhteline asukoht on
0,500 ning 50% näidetest jääb vahemikku 0,485–0,516. Kõik need pildid on
rõhtpaigutusega, mediaanse küljesuhtega 1,414. Kontrollitud näites tähistab
`<pb/>` korrektselt kõrvuti skaneeritud vasaku ja parema lehekülje piiri.
See annab mudelile väga tugeva ja järjekindla visuaalse õpisignaali ning
selgitab hästi, miks mudel märgendab `<pb/>` juba edukalt.

Loendus näitab esinemissagedust, mitte märgenduse kvaliteeti. Haruldased
märgendid ei ole praegu peamine probleem: `<fn>` ei ole sisuliselt kasutusel,
`<b>` esineb ainult erandjuhtudel ning `<noodid>` on küll harv, kuid visuaalselt
lihtne marker, mille mudel õppis ära juba few-shot näidetest. Nende sageduse
kunstlik suurendamine ei ole praegu põhjendatud.

Kui tähetranskriptsioon ja tavaline üheveeruline tekst on juba head, annab
juhuslikult suurema puhta teksti andmestiku lisamine vähe. Arendus peab
keskenduma päriselt olulistele ja rasketele märgendamisotsustele:

- `<i>` ja `<cs>` visuaalne eristamine ning nende täpsed piirid;
- `<m>` ploki eristamine põhitekstist ja õige paigutamine väljundisse;
- sama lehe sees nii märgendatud kui märgendamata sarnase välimusega tekst;
- marginaali ja põhiteksti ebaselge ruumiline suhe;

ning keeruka küljendusega lehti, kus esinevad:

- kaks või enam veergu;
- marginaalid mõlemal küljel;
- põhiteksti katkestavad päised või jalused;
- suur tühi ruum ja mitu eraldi tekstiplokki;
- segamini Antiqua, Fraktur ja kreeka kiri;
- topeltleheküljed ning `<pb/>`;
- väga pikad transkriptsioonid.

Raskete näidete oversampling peab jääma mõõdukaks, et mudel ei hakkaks nägema
iga lehte keeruka erandina. Samuti tuleb auditeerida, kas ground truth'is on
märgendamisotsused eri teoste vahel järjekindlad. Mudel ei saa õppida
stabiilset visuaalset reeglit, kui sama nähtus on kord märgendatud ja kord
märgendamata.

Praeguses 1066-lehelises treening-CSV-s on `<m>` vorming ise ebaühtlane:

- 6847 avavat ja 6836 sulgevat `<m>` märgendit;
- 6205 üherealist märgendipaari;
- 615 mitmerealist märgendipaari 239 lehel;
- 4 lehte olid tasakaalustamata `<m>` märgenditega.

Kõik neli vigast lehte pärinesid samast teosest
`1636-9-Oratio_de_castitate...` (lehed 0018, 0020, 0021 ja 0022). Probleem
ei olnud keerukas ristuv pesastus: plokivormilt reavormile üleminekul olid
mõne marginaali ette jäänud üleliigsed avajad, näiteks
`<m><m>Ratio 2.</m>` ja ühel juhul `<m><m><m>Ratio 3.</m>`. Piltide järgi on
need tavalised eraldi marginaaliread, mitte pesastatud marginaalid.
Normaliseerija eemaldab nüüd vahetult teise `<m>` ees olevad vigased avajad.
Täiendav audit leidis ka sama tagi sisemist topeltpesastust, näiteks
`<m>3<m>.</m></m>`, `<i><i>X</i></i>` ja pesastatud `<cs>` lõike. Need
lamendatakse sisu muutmata üheks samaliigiliseks märgendiks. Parandada tuli ka
normaliseerija juhtum, kus igal real juba olemas olev `<i>...</i>` võis saada
teise `<i>` paari ümber.

Pärast kogu puhastusahela rakendamist on CSV-s 7782 avavat ja 7782 sulgevat
`<m>` märgendit, null mitmerealist `<m>` paari ning null tuvastatud sama tagi
topeltpesastust. Ühel real puudunud `<m>` avaja parandatakse automaatselt.
Puhastus käib püsipunktini ühises `clean_markup()` funktsioonis, mistõttu
andmestiku ehitaja ja treener annavad nüüd identsed tulemused. Tundmatuid
märgendeid ega puuduvaid pilte ei ole.

Prompt nõuab samal ajal, et mitmerealise marginaali iga rida oleks eraldi
`<m>` märgendis. Kanooniliseks vormiks valiti seetõttu üks `<m>` paar iga
marginaalirea ümber. `normalize_multiline_m_tags()` rakendub nüüd nii
`build_vutt_dataset.py` kui ka `train_markup.py` puhastusahelas; VUTT-i
originaalfaile ei muudeta. Vana CSV normaliseeritakse treeneris jooksvalt ja
järgmine andmestiku ehitamine kirjutab juba normaliseeritud CSV.

Failide muutmisaeg toetab kasutaja oletust osaliselt, kuid mitte absoluutselt.
`vutt_sync.py` kasutab `rsync -a`, seega säilib serveri mtime, kuigi 5. märtsi
ühesugused ajatemplid võivad tähistada ka massimporti, mitte tegelikku
märgendamise kuupäeva.

| Toorfaili mtime | Üherealisi `<m>` paare | Mitmerealisi paare | Mitmerealiste osakaal |
|---|---:|---:|---:|
| 2026-03 | 1324 | 297 | 18,3% |
| 2026-04 | 475 | 5 | 1,0% |
| 2026-05 | 229 | 10 | 4,2% |
| 2026-06 | 2078 | 95 | 4,4% |
| 2026-07 | 2132 | 210 | 9,0% |

Märtsis on mitmerealine vana vorm selgelt tavalisem ja ainult mitmerealist
vormi sisaldavate lehtede mediaanne mtime on 5. märts. Siiski leidub
mitmerealisi plokke ka juulis ning 223 lehel on mõlemad vormid korraga.
Seega on suundumus olemas, kuid kasutajate sisestusviis pole ka hiljutistes
andmetes täielikult ühtlustunud.

#### Juhtum: `1626-...-lzogr0-146.jpg`

See testleht selgitab hästi, miks `<m>` on raskem kui `<pb/>` või `<noodid>`:

- leht on väga suur, 7155 × 8525 ehk 61 MP, ning 5,12 MP eelarve vähendab
  selle umbes 2072 × 2469 pikslile (lineaarne mõõt 29% originaalist);
- paremas servas on mitu eri kõrgusel marginaaliplokki, mitte üks lihtne
  marker;
- marginaal on kitsas, väiksemas kirjas ja kohe põhiteksti kõrval;
- marginaaliplokid tuleb mitte ainult ära tunda, vaid transkribeerida,
  ridadeks jagada ja siduda põhiteksti õige kohaga;
- Fraktuuris põhiteksti kõrval olev Antiqua marginaal tekitab korraga nii
  ruumilise `<m>` kui kirjatüübi/koodivahetuse küsimuse.

Viimane punkt vajab märgendusreeglis selget otsust. Sama teose valmis lehel
021 on põhiteksti Antiqua sõnad märgendatud `<cs>`-iga, kuid Antiquas
marginaal on ainult `<m>` sees, ilma `<cs>`-ita. Kui reegel on, et `<m>`
kirjeldab marginaali ja selle sees kirjatüüpi eraldi ei märgendata, tuleb see
prompt'is sõnaselgelt öelda; vastasel juhul on mudeli jaoks kaks kattuvat
õiget klassifikatsiooni.

Lehe VUTT toorfail on staatusega `Toores` ja selle transkriptsioonis pole
`<m>` märgendeid; marginaalitekst on lihtsalt põhiteksti järel. Seega ei ole
see fail veel kasutatav automaatse markup-hindamise ground truth'ina. Sellele
tuleb koostada käsitsi kanooniline testvastus.

Sama teose 178 toorfailist on ainult kaks staatusega `Valmis`; mõlemad on
praeguses treeningandmestikus ja sisaldavad `<m>` näiteid. Leht 021 on
visuaalselt sarnane marginaalidega näide, kuid seal on marginaal vasakul ja
selgemalt eraldatud. Üks-kaks sama teose näidet ei kata lehe 146 parempoolse,
tiheda ja mitmeplokilise marginaali keerukust.

Mudelite senised väljundid näitavad eri tüüpi ebastabiilsust: 20260527 ja
20260531 leidsid marginaalid ning koondasid need lehe lõppu; 20260614 jagas
need üherealisteks `<m>` elementideks, kuid paigutas osa liiga vara ja kordas
kolme viidet; 20260720 ja 20260721 jätsid marginaalid üldse välja. See viitab
pigem `<m>` väljundformaadi ja õppeandmete ebastabiilsusele kui üldisele
võimetusele tähti lugeda.

### Katse C: trükimudeli LoRA rank 16 versus 64

Praegune trükimudel kasutab rank 16 adapterit. Käsikirjamudeli varasem katse
näitas, et rank 16 tulemused olid üsna kesised ja rank 64 andis parema
tulemuse. Seetõttu ei ole rank'i võrdlus enam pelgalt teoreetiline, vaid üks
prioriteetsemaid 9B trükimudeli katseid.

Põhivõrdlus võiks olla:

- praegune r=16 kontrollmudel;
- uus r=64 mudel samast 9B baasmudelist, sama andmejaotuse ja sama sammude
  arvuga;
- soovi korral r=32 vahepunkt, kui on vaja leida odavam kompromiss.

Hinnata tuleb eraldi puhast tähetranskriptsiooni ja markup'i. Võimalik on, et
r=64 ei muuda CER-i peaaegu üldse, kuid parandab märgendite liike, piire ja
ruumilisi seoseid — just see oleks praeguse probleemi puhul edukas tulemus.

Olemasoleva rank 16 checkpoint'i rank'i ei saa lihtsalt jätkamisel muuta.
Ausaks võrdluseks tuleb luua uus adapter samast algsest baasmudelist ja läbida
sama treeninguetappide järjestus.

### Katse D: vision encoder'i kohandamine

Praegu lisatakse LoRA nii vision- kui language-kihtidele. Võrrelda tuleks:

1. vision külmutatud, LoRA ainult language/projection osal;
2. vision LoRA + language LoRA nagu praegu;
3. vajaduse korral ainult vision-projektori ja language-kihtide LoRA.

Vision encoder'i täielik kohandamine ei ole automaatselt parem. Väikese või
ühetaolise OCR-andmestiku korral võib see üldist visuaalset võimekust
halvendada; ajalooliste kirjatüüpide puhul võib kohandamine samas olla väga
vajalik. Otsus peab tulema valideerimistulemusest.

### Katse E: kaheetapilise treeningu unustamise kontroll

Praegune markup-treening jätkab baas-OCR adapterist, kuid treenib ainult
VUTT markup-andmetel. Tuleb kontrollida, kas markup paraneb puhta OCR-i,
kreeka teksti või haruldaste allikate arvelt.

Võimalik lahendus on rehearsal-mix, näiteks:

- 70–90% VUTT markup-lehti;
- 10–30% esimese etapi puhta transkriptsiooni lehti.

Proportsioon tuleb valida valideerimisega. Ka markup'i õppemäära `1e-4` ja
ainult kümmet warmup-sammu tasub võrrelda väiksema õppemäära ning
`warmup_ratio`-ga.

### Katse F: assistant-only loss'i kontroll

Tuleb kontrollida, kas praegune `UnslothVisionDataCollator` ja `SFTTrainer`
arvutavad loss'i ainult assistendi vastusel või ka kasutaja pikal
instruktsioonil. OCR-treeningus on üldjuhul soovitav, et peamine loss tuleks
transkriptsioonilt. Seda ei tohi muuta oletuse põhjal; esmalt tuleb vaadata
tegelikke `labels` maske ühe batch'i peal.

### Katse G: mitmeskaalaline lehekülg

Kui üks täislehekülje pilt ei anna korraga piisavalt head küljendust ja
kirjatäpsust, võib hiljem katsetada sisendit, kus mudel näeb:

- kogu lehekülje vähendatud pilti küljenduse jaoks;
- lisaks üht või mitut suurema resolutsiooniga lõiget teksti jaoks.

See nõuab eraldi treeningnäite formaati ja suurendab visuaaltokenite arvu.
Seetõttu ei ole see esimene katse. Lihtne lehe tükkideks lõikamine võib
parandada tähetäpsust, kuid kaotada just selle tervikstruktuuri, mida soovime
parandada.

## 27B mudelile liikumine

Põhistrateegia jääb:

1. arendada ja valideerida toru 9B peal lokaalselt;
2. rentida 27B piloodiks suure mäluga NVIDIA GPU;
3. mõõta VRAM-i, kiirust ja kvaliteedivõitu;
4. teha täistreening pilves ainult juhul, kui 27B annab raskel
   valideerimiskorpusel selge paranemise.

Praeguseid skripte ei saa 27B jaoks kasutada ainult mudelinime vahetades:

- `scripts/train.py` on seotud `unsloth/Qwen3.5-9B` mudeliga;
- `scripts/train_markup.py` eeldab juba küljes olevat 9B LoRA adapterit;
- 9B LoRA kaale ei saa 27B mudelile üle kanda;
- merge'i ega Q5/Q6 kvantimise toru pole praegu olemas.

Üldistatud treeningskript peaks toetama vähemalt argumente:

```text
--model
--adapter
--type print|hand
--max-pixels
--max-length
--lora-rank
--freeze-vision
--learning-rate
--steps
--resume
--output
```

Uue 27B baasmudeli puhul tuleb `get_peft_model()` kutsuda täpselt üks kord;
olemasoleva 27B adapteri jätkamisel ei tohi seda kutsuda.

Esimene 27B piloot on mõistlik teha 80 GB GPU-l. Praegune umbes 5000
visuaaltokeni, 8192 kogupikkuse ning vision+language LoRA konfiguratsioon
võib 48 GB peal olla väga piiripealne. 80 GB jooks näitab tegelikku varu ja
selle põhjal saab otsustada, kas 48 GB kaart on täistreeninguks realistlik.

## 27B lokaalne inferents

27B BF16 ei mahu 32 GB VRAM-i. Q5/Q6 kaalude teoreetiline maht võiks mahtuda,
kuid praegune kood toetab otseselt ainult BF16 ja `load_in_4bit=True` teed.
Q5/Q6 kasutatavus sõltub sellest, kas valitud GGUF/GPTQ/AWQ runtime toetab
konkreetselt Qwen3.5 multimodaalset arhitektuuri ja selle vision-osa.

Seetõttu tuleb enne 27B täistreeningut kontrollida ka juurutusteed. Valmis
mudelist pole kasu, kui soovitud kvantimisformaat ei toeta pilte või annab
liiga aeglase inferentsi. 27B lokaalsel testimisel tuleb alustada batch
size'iga 1; praegune `scripts/test_model.py` kasutab batch size'i 3.

## Esialgne tegevusjärjekord

1. Kontrollida automaatselt normaliseeritud `<m>` andmeid enne iga uut
   treeningut.
2. Koostada lehele `1626-...-146` ning teistele rasketele marginaalilehtedele
   käsitsi kontrollitud markup-ground-truth.
3. Moodustada fikseeritud markup-valideerimiskorpus ja mõõta eraldi puhast
   teksti ning märgendeid.
4. Teha tokenipikkuste audit ja kontrollida `labels` maske.
5. Parandada print/hand prompt'i valik testskriptis.
6. Võrrelda 9B trükimudelil ausalt rank 16 ja rank 64 adapterit.
7. Lisada teosepõhine valideerimisjaotus ning raske küljenduse testkorpus.
8. Vajaduse korral ülekaalustada keerukate `<m>` plokkidega lehti.
9. Võrrelda vision encoder'i külmutamist ja rehearsal-mix'i.
10. Muuta checkpoint-resume ja logimine töökindlaks.
11. Alles seejärel üldistada skript 27B piloodi jaoks.
12. Teha 27B 500–1000 sammu piloot 80 GB GPU-l ja võrrelda sama fikseeritud
    raske testkorpusega.

## Lahtised küsimused

- Milline kolmest põhimärgendist eksib kõige sagedamini: `<m>`, `<i>` või
  `<cs>`?
- Kas viga seisneb märgendi liigis, märgendi piirides või kogu tekstiploki
  vales rollis?
- Kas terviklehekülje vead esinevad peamiselt pikkadel lehtedel või kindlatel
  küljendustel?
- Kas lisaks markup-veale esineb teksti puudumist, vale lugemisjärjekorda või
  hallutsineeritud kordust?
- Kui suur osa treeningnäidetest ületab 8192 tokenit?
- Kas praegune loss on ainult assistendi väljundil?
- Kas vision LoRA parandab rasket küljendust võrreldes külmutatud visioniga?
- Kas markup-etapp halvendab esimese etapi puhast OCR-i?
- Kas rank 32/64 annab 9B puhul mõõdetava võidu?
- Milline kvantimisruntime toetab hiljem 27B mudeli vision-osa ja Q5/Q6
  formaati?

Peamine tööhüpotees on, et 9B baasmudel võib olla ülesande jaoks piisav ning
praegune piirang võib asuda rank 16 adapteris või markup'i õppeandmetes, mitte
tähemärkide visuaalses äratundmises. Esimene sisuline mudelikatse peaks olema
rank 64 trükimudel koos eraldi markup-mõõdikutega. 27B mõte tuleks uuesti
hinnata alles pärast neid katseid.
