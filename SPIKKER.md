# OCR mudeli treenimise spikker

## Temperatuuri ja GPU jälgimine

```bash
# CPU temperatuurid (°C)
paste <(cat /sys/class/thermal/thermal_zone*/type) \
      <(awk '{printf "%.0f\n", $1/1000}' /sys/class/thermal/thermal_zone*/temp) \
      | column -s $'\t' -t

# GPU temperatuur ja koormus
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader

# Pidev jälgimine (2s interval)
watch -n2 "paste <(cat /sys/class/thermal/thermal_zone*/type) \
  <(awk '{printf \"%.0f\n\", \$1/1000}' /sys/class/thermal/thermal_zone*/temp) \
  | column -s \$'\t' -t; echo; \
  nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader"
```

## Pikaks treeninguks — pärast reebooti seadista uuesti!

```bash
sudo nvidia-smi -pl 450                                              # GPU 450W (vaikimisi 575W)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo     # CPU turbo välja

# SSH võti agenti (küsib parooli korra) – ilma selleta ei tööta
# git push ega vutt_sync.py rsync automaatselt
SSH_AUTH_SOCK=/run/user/1000/openssh_agent ssh-add ~/.ssh/id_ed25519
```

Kontroll, kas võti on agendis:

```bash
SSH_AUTH_SOCK=/run/user/1000/openssh_agent ssh-add -l
```

`The agent has no identities` = võti on laadimata, ssh hakkab parooli küsima.

---

## SSH kaudu käivitamine — kasuta alati tmux-i!

```bash
tmux new -s treening          # uus sessioon (tee seda enne kõike muud)
# Ctrl+A, D                   # lahku sessioonist (treening jookseb edasi)
tmux attach -t treening       # tule tagasi uue SSH sessiooni järel
tmux ls                       # vaata aktiivseid sessioone
```

---

## Täistreenimine (kõik sammud korraga)

```bash
cd /home/mf/Dokumendid/LLM/qwen3.5
source venv/bin/activate
bash scripts/train_pipeline.sh
```

See teeb järjest: VUTT sünk → andmestiku ehitamine → ocr-service peatus → treenimine → ocr-service käivitus.

Testjooks (ei salvesta, ~6 min):
```bash
bash scripts/train_pipeline.sh --test
```

---

## Sammud käsitsi

### 1. VUTT andmed — tulevad backupist, sünkima ei pea

Öine cron (`vutt_backup.py`, VUTT repost) tõmbab VUTT serveri tervikuna
snapshot'i `~/vutt-backups/latest/data`. Eraldi sünki EI OLE vaja.

```bash
ls -l ~/vutt-backups/latest                      # millise snapshot'i peale osutab
journalctl -t vutt-backup --since today          # kas öine jooks õnnestus
```

`scripts/vutt_sync.py` on aegunud (jooksis ilma `--delete`-ita → serverist
kustutatud lehed jäid alles). Hoia varuvariandiks, kui backup-masin on maas.

### 2. Andmestiku ehitamine
```bash
python scripts/build_vutt_dataset.py --stats   # statistika
python scripts/build_vutt_dataset.py           # kirjuta failid
python scripts/build_vutt_dataset.py --type hand    # käsikirjad (vaikimisi print)
python scripts/build_vutt_dataset.py --raw-dir /muu/tee   # muu allikas
```
Tulemus: `data/vutt/metadata.csv` + `data/vutt/images/` + `data/vutt/SOURCE.txt`

`SOURCE.txt` ütleb, MILLISEST snapshot'ist see andmestik tehti — kuu aega hiljem
on see ainus viis teada, mille peal mudel treeniti.

Trüki- ja käsikirjamudelit treenitakse eraldi, seega `--type` filtreerib
teose `_metadata.json` järgi (Wikidata `Q1261026` = trükis, `Q87167` =
käsikiri). Kui statistika kurdab puuduva `type` välja üle, tuleb see
VUTT-is ära täita — muidu jäävad need teosed vaikselt välja.

**Puhastusahel — transkriptsiooni muudetakse automaatselt:**

| Samm | Mida teeb |
|---|---|
| `unwrap_tags` | `<ann1>`–`<ann4>` märgend maha, sisu alles |
| `fix_crossed_tags` | ristuv pesastus `<i>..<cs>X</i></cs>` → `<i>..<cs>X</cs></i>` |
| `normalize_multiline_m_tags` (1) | mitmerealine `<m>A\nB</m>` → `<m>A</m>\n<m>B</m>`; eemaldab ka vigased pesastatud avajad `<m><m>A</m>` |
| `flatten_redundant_nested_tags` | sama tagi topeltpesastus `<m>3<m>.</m></m>` → `<m>3.</m>` ja `<i><i>X</i></i>` → `<i>X</i>` |
| `normalize_multiline_m_tags` (2) | jagab uuesti plokid, mille vigase pesastuse lamendamine muutis mitmerealiseks |
| `balance_line_m_tags` | lisab üksikul marginaalireal puuduva `<m>` avaja või sulgeja |
| `remove_empty_m_tags` + `remove_empty_tags` | tühjad märgendipaarid välja |
| `clean_markup` | kordab eelnevat ahelat püsipunktini; ehitaja ja treener kasutavad sama funktsiooni |
| tühja teksti kontroll | 0-baidised "Valmis" lehed jäävad välja |

Sama ahel jookseb ka `train_markup.py` laadimisel, et vanemad CSV-d samuti
puhastuks. **NB!** See tähendab, et treeningandmed ei ole bait-bait samad
mis VUTT-is — kui mudeli väljundis midagi kummalist paistab, pea seda meeles.

Avajata sulgejaid ja sulgejata avajaid EI parandata: need on leheküljepiiri
ületavad jooksud (kaldkiri algab eelmisel lehel) ja on õiguspärased.

Lehed, mida parandus ainult vormistab ja mis vajavad VUTT-is käsitsi
parandust (liigne `<m>` keset sõna, dubleeritud `<i>`), on loetletud failis
`docs/arhiiv/katkised-lehed-20260721.txt`.

### Tühjad ja hõredad leheküljed treeningandmetes

**Seis 26.08.2026:** käsikirjapool on tehtud – `data/kurrent/metadata.csv`
sisaldab 23 `[tühi lehekülg]` ja 3 hõredat lehte (allikas `vutt_horedad`).
Trükipool (`data/vutt`) on tegemata. Praktikas selgus, et **ainuüksi prompti
täiendus lõpetas loopi jooksmise tühjadel lehtedel** – treeningnäited on
pigem kindlustus, et uus peenhäälestus seda käitumist ära ei nulliks.
Allpool on probleemi algne kirjeldus ja nõuded, mis kehtivad edasi.


**Probleem:** kui mudelile anda tühi (või peaaegu tühi) lehekülg, läheb ta
loopi ja genereerib maksimumini täiesti suvalist teksti. Tagajärg on
kahtlaselt vastupidine intuitsioonile: **tühi lehekülg võtab praegu
tunduvalt rohkem aega kui tekstiga lehekülg** — tekstiga leht lõpetab
loomulikult EOS-iga, tühi leht jookseb iga kord token-lakke (~8 min lehe
kohta). Kui partiis on tühje lehti, on aeglus just nende taga, mitte
raske teksti taga.

Põhjus on lihtne: **mudel ei ole kunagi näinud ühtegi näidet, kus õige
vastus on „lehekülg on tühi"**, seega pole tal midagi, mille peal lõpetada.

**Sama probleemi teine pool: hõredad lehed.** Mudel ei ole näinud ka lehti,
kus on lihtsalt *vähe* teksti – ainult leheküljenumber, pealkiri, kolofon,
paar lõpurida. Ta on õppinud, et lehel peab teksti palju olema, ja hakkab
hõredal lehel puuduolevat juurde luuletama. Tühi lehekülg on selle skaala
äärmus, mitte eraldi nähtus – seepärast lahendatakse mõlemad koos.

Tekstipikkuse jaotus (mõõdetud 18.08.2026, märgendid maha arvatud):

| Andmestik | Mediaan | ≤50 märki | ≤100 | ≤200 |
|---|---|---|---|---|
| `data/vutt` (markup, trükk) | 2254 | **0** | **0** | 5 |
| `data/kurrent` | 1338 | 7 | 139 | 296 |
| `data/lehekyljed` (etapp 1) | 993 | 17¹ | 17 | 36 |

¹ neist 15 on tühjad, mis treeningust välja filtreeritakse.

Trükimudel ei ole seega näinud **ühtegi** lehte alla 100 märgi; kõige
hõredamad, mida ta teab, on tiitellehed (~101 märki). Kurrent-poolel on
hõredaid lehti olemas, aga alla protsendi.

**Kontrollitud 18.08.2026 – ükski andmestik ei sisalda tühja lehekülge:**

| Andmestik | Ridu | Tühja transkriptsiooniga |
|---|---|---|
| `data/vutt/metadata.csv` (markup, trükk) | 1113 | 0 |
| `data/kurrent/metadata.csv` | 17 018 | 0 |
| `data/kurrent/metadata_balanced.csv` | 11 018 | 0 |
| `data/processed/metadata.csv` | 136 | 0 |
| `data/lehekyljed/metadata.csv` (etapp 1) | 1500 | 15 – **aga need filtreeritakse treeningust välja** |

Ka „peaaegu tühje" lehti (alla 20 tähemärgi) ei ole üheski andmestikus.
Nullid ei ole juhus, vaid tulevad kahest filtrist:
`build_vutt_dataset.py` viskab tühja transkriptsiooniga lehed välja
(loendur `Vahele jäetud (tühi tekst)`) ja `train.py:125` teeb sama laadimisel.
Seega **mitte ükski senine treening – ei etapp 1, ei markup, ei Kurrent –
ei ole näinud ühtegi tühja lehekülge.**

Üks kasulik leid: need 15 rida failis `data/lehekyljed/metadata.csv` on
päris tühjade lehtede ja köidete pildid (Gezeliuse eeslehed 0002–0006 ja
0445–0448, köitepildid `eb01`–`eb04`, `00140`, kaks disputatsiooni
vahelehte) – **pildid on kettal alles**, ainult transkriptsioon on tühi
string. Trükipoole tühjade lehtede stardikomplekt on seega juba olemas:
piisab, kui asendada tühi string stringiga `[tühi lehekülg]`. NB! `eb01`–
`eb04` on köitekaaned (mõõteskaala ja raamatukogu silt pildil), mitte
tühjad leheküljed – need on eri juhtum ja väärivad eraldi otsust.

**Kokku lepitud transkriptsioon** (otsustatud 18.08.2026) – tühja lehe ainus
sisu on täpselt üks rida:

```
[tühi lehekülg]
```

Sama string peab käima läbi kolme koha: VUTT-i märgendus, `scripts/prompt.py`
juhis ja andmestiku ehitajad. Vabateksti variante („tühi", „leht on tühi")
ei tohi tekkida – muidu ei õpi mudel seda kui lõpetamismärki.

Ülejäänud nõuded:

- **Kogus ~1–3 % lehtedest** – piisav, et muster kinnistuks, aga mitte nii
  palju, et mudel hakkaks tühjust ka kirjutatud lehtedele pakkuma.
- **Hõredad lehed** (ainult leheküljenumber, signatuur, pealkiri, paar rida,
  tint-plekk) **ei ole tühjad** – need transkribeeritakse tavaliselt, täpselt
  see, mis lehel on. Neid on treeningandmetesse vaja sama moodi nagu tühje:
  ilma nendeta ei tea mudel, et lühike vastus võib olla õige vastus. Leht,
  millel on ainult „42", saab transkriptsiooniks `42`.
- Ehitaja tühja-teksti filter jääb alles: `[tühi lehekülg]` **ei ole** tühi
  string, seega ta läbib filtri probleemideta. Päriselt 0-baidised lehed
  jäävad edasi välja (need on märgendamata, mitte tühjad).
- Seni kuni see puudu on, tasub inferentsi poolel hoida käes ka
  n-o hädapidur: token-lakke jooksnud väljundi äratundmine ja äraviskamine.

**Trükilehed – lahendus olemas.** Tühjad trükileheküljed märgendatakse
VUTT-is käsitsi ja tulevad tavalist teed pidi: backup-snapshot →
`build_vutt_dataset.py` → `data/vutt/metadata.csv` → `train_markup.py`.
Eraldi lippe ei ole vaja – tavaline jooks korjab tühjad lehed koos muuga
ja kontrollib märgendi vormi. Seisuga 18.08.2026 on VUTT-is juba **12 Valmis
lehte tühja transkriptsiooniga** (6 trükis + 6 käsikirjas) – need on esimesed
kandidaadid, mida `[tühi lehekülg]`-ks märkida. `--stats` loetleb nad
teose/lehe kaupa ette.

**Käsikirjad – suurem probleem, aga tee on valitud.** Kurrent-mudel
treenitakse failist `data/kurrent/metadata.csv` (17 018 rida, 12 välisallikat
+ 2039 määramata; kontrollitud 18.08.2026: **0 tühja lehekülge**). Need on
arhiivikorpused (Riksarkivet, Bullinger, Hanse, Dresden, senatsprotokollid) –
neid ei saa VUTT-is märgendada ja tühje lehti nad ei sisalda, sest
transkribeeritud on ainult kirjutatud leheküljed. Praktikas on käsikirjade
tühjad lehed sagedasemad kui trükistel (tühjad versopooled, eeslehed),
seega häirib probleem siin rohkem.

**Otsustatud (18.08.2026):** tühjad käsikirjaleheküljed märgendatakse samuti
VUTT-is (`type=Q87167` teosed, transkriptsioon `[tühi lehekülg]`, staatus
Valmis) ja **lisatakse olemasolevale 17 000 lk andmestikule** uue allikana –
mitte eraldi paralleelse andmestikuna. `data/kurrent/metadata.csv` on
3-veeruline (`failinimi, transkriptsioon, allikas`), seega rida on lihtne
juurde panna: allikanimi nt `vutt_tyhjad`, ja `scripts/filter_dataset.py
--max-per-source` hoiab osakaalu paigas.

**Siinpoolne osa on valmis (18.08.2026)** – jäänud on ainult VUTT-is
märgendamine:

- `scripts/prompt.py`: konstant `EMPTY_PAGE_MARKER = "[tühi lehekülg]"` on
  ainus tõeallikas; mõlemasse juhisesse – `INSTRUCTION` (trükk/markup) ja
  `KURRENT_INSTRUCTION` (käsikiri) – on lisatud kaks reeglit: tühja lehe
  märgend ja „hõredat lehte ei tohi täis luuletada".
- `build_vutt_dataset.py`: uued lipud `--out`, `--append`, `--allikas`,
  `--only-empty`, `--max-chars`, `--force` + märgendikontroll ja hõredate
  lehtede aruanne.

**Käsikirjade tühjade ja hõredate lehtede lisamine 17 000 lk andmestikule:**

```bash
# 1. Vaata üle, mis VUTT-ist tuleb (ei kirjuta midagi)
python scripts/build_vutt_dataset.py --type hand --max-chars 100 --stats

# 2. Lisa Kurrent-andmestikule (tühjad + hõredad ühe käiguga)
python scripts/build_vutt_dataset.py --type hand --max-chars 100 \
    --out data/kurrent --append --allikas vutt_horedad
```

`--max-chars N` võtab lehed, mille tekstis on kuni N märki (XML-märgendid
maha arvatud) – tühjad lehed mahuvad alati sisse, sest märgend ise on 15
märki. Ainult päris tühje lehti annab `--only-empty`. Kui mõlemad lipud on
korraga, jäävad alles ainult tühjad.

Ilma valikuliputa jooks **loendab ja loetleb** hõredad lehed (≤100 märki),
aga ei filtreeri midagi – nii on kohe näha, kas neid on üldse.

Mida kontroll teeb:

| Olukord | Tulemus |
|---|---|
| täpselt `[tühi lehekülg]` (ümbritsev tühik lubatud) | läheb andmestikku, salvestatakse kanoonilises vormis |
| `tühi leht`, `[Tühi lehekülg]`, `blank page`, `vacat`, täpitähtedeta variant | **jääb välja**, loetletakse „VALES VORMIS" all – paranda VUTT-is |
| `[tühi lehekülg]` koos muu tekstiga | **jääb välja**, loetletakse eraldi (kas leht pole tühi või jäi märgend sisse) |
| Valmis leht tühja `.txt`-ga | jääb nagu enne välja, aga skript ütleb, et äkki peaks olema märgitud tühjaks |

Ohutuslukud, mis samal ajal tekkisid:

- `--type hand|all` **ilma `--out` liputa annab vea** – muidu kirjutaks
  käsikirjajooks üle `data/vutt/` trükiandmestiku.
- `--append` ei kirjuta olemasolevat CSV-d ümber, vaid lisab read lõppu;
  juba olemas olevad failinimed jäetakse vahele (kordusjooks on ohutu).
  Kolmas veerg `allikas` täidetakse `--allikas` väärtusega.
- Ilma `--append` liputa kolmeveerulise CSV peale kirjutamine on **keelatud**
  (nõuab `--force`) – muidu hävitaks üks käsklus 17 000 rida.
- `SOURCE.txt` saab `--append` puhul uue kirje juurde, vana jääb alles.

**Kogus:** 17 018 rea juures tähendab 1–3 % umbes 170–500 tühja lehte. Nii
palju käsitsi märgendada pole mõtet – alusta väiksemast (30–50) ja vaata,
kas mudel hakkab tühja lehe peal lõpetama; vajadusel korda.

### Piltide eelskaleerimine (`--resize`) – lugege enne kasutamist

Meie skaneeringud on mediaanis ~15 MP, eelarve on 5,12 MP. Vaikimisi läheb
täissuuruses pilt kettalt protsessorisse, mis skaleerib ta **igal epohhil
uuesti** – mõõdetuna 310 ms lehe kohta, ja see paneb GPU pauside ajal
seisma. `--resize` teeb selle töö ühe korra ära, ehitamise ajal:

```bash
python scripts/build_vutt_dataset.py --resize
```

Mõõdetud võit: **310 ms → 31 ms lehe kohta (~10x)**, maht 1,82 GB → ~75%.

**Aga see seob andmestiku inferentsiga.** Eelskaleerimine kasutab LANCZOS-i,
protsessor BICUBIC-ut; nende vahe on 42–47 dB PSNR ehk sama suurusjärk mis
JPEG-i ümberkodeerimine. Kui treening näeb üht ja inferents teist, on tegu
treening/inferents-nihkega.

Seega **`--resize` andmestikul treenitud mudeli aktiveerimisel tuleb SAMAL
AJAL** lisada `imaging.fit_to_budget()` kutse enne protsessorit nii failis
`kataloogi-jalgimine-ja-ocr.py` kui `scripts/test_model.py`. Mõlemas on
kommentaar täpses kohas.

Eelarve `MAX_PIXELS` elab ühes kohas: `scripts/imaging.py`.

| Mudel | Andmestik | Inferents peab olema |
|---|---|---|
| kuni `20260721` (k.a) | täissuuruses | nagu praegu (protsessor skaleerib) |
| `--resize` andmestikuga treenitud | eelskaleeritud | `fit_to_budget()` enne protsessorit |

### 3. Treenimine
```bash
sudo systemctl stop ocr-service          # vabasta GPU mälu!
python scripts/train_markup.py --test    # testjooks
python scripts/train_markup.py           # täistreening (~2-4h)
sudo systemctl start ocr-service         # taaskäivita pärast
```
Tulemus: `models/qwen3.5-ocr-markup-YYYYMMDD/`

---

## Kurrent-treening (käsikiri) — töökäik algusest lõpuni

Seis 26.08.2026: **andmestik ja tööriistad on valmis, jooks ootab
käivitamist** (plaanitud reede 28.08 pärastlõunal, kestab nädalavahetuse).
Trükimudelit see ei puuduta – Kurrent on eraldi mudel.

### Enne käivitamist

| # | Käsk | Mida oodata |
|---|---|---|
| 1 | `sudo systemctl stop ocr-service llama-server-print llama-server-hand` | **kolm teenust**, kokku ~25,5 GB |
| 2 | `nvidia-smi --query-gpu=memory.used --format=csv,noheader` | alla 1000 MiB |
| 3 | `sudo nvidia-smi -pl 450` | lähtestub iga reboodiga |
| 4 | `echo 1 \| sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo` | CPU 77 °C → 55 °C |
| 5 | `tmux new -s kurrent` | 33-tunnine jooks ei tohi SSH otsa surra |
| 6 | `venv/bin/python scripts/train_kurrent.py --test` | 5 sammu, ~5 min |

Testjooksu väljundis PEAB seisma täpselt see:

```
Lähtepunkt:    unsloth/Qwen3.5-9B
LoRA rank: 64
  Holdout: 133 lehte treeningust välja (data/kurrent/holdout.txt)
  Andmestik: 18775 näidet
```

Kui lähtepunkt on mõni `models/...` või rank 16 – **peatu**, vaikeväärtused on
jälle valed ja tulemuseks oleks hoopis teine mudel (vt commit a02a86c).

### Täisjooks

```bash
venv/bin/python scripts/train_kurrent.py 2>&1 | tee /tmp/kurrent-treening.log
```

Ootused: **2122 sammu epohhis, 2 epohhi = 4244 sammu, ~27,9 s/samm ≈ 33 h.**
Väljund `models/qwen3.5-ocr-kurrent-YYYYMMDD` (kuupäev = käivitamise päev),
checkpointid `models/checkpoints-kurrent-YYYYMMDD/`.

### Checkpointid ja katkemine (muudetud 29.08.2026)

`save_strategy="epoch"` tähendas, et esimene päästerõngas tekkis alles **12,3 h
pärast** – crash 11. tunnil kaotas kõik. Nüüd:

| säte | väärtus | tähendus |
|---|---|---|
| `save_steps` | 250 | checkpoint ~1,9 h tagant |
| `save_total_limit` | 3 | 3,6 GB, vanemad roteeruvad välja |
| `epohh-N-adapter/` | iga epohhi lõpus | ~800 MB, **rotatsioonist väljas** |

Epohhi lõpu adapter on hindamiseks, mitte jätkamiseks (optimeerija olekut ei
sisalda). Ta on vajalik selleks, et hiljem saaks võrrelda 1. ja 2. epohhi –
20260602 jooksul andis teine epohh vähe (loss 0,12 → 0,08).

**Katkemise järel jätkamine:**

```bash
venv/bin/python scripts/train_kurrent.py --resume
```

`--resume` ilma argumendita leiab viimase `models/checkpoints-kurrent-*` kausta
ja suurima `checkpoint-N` selles. **Ta jätkab ka VANA kuupäevatempliga** – jooks
kestab üle südaöö, ja ilma selleta tekiks `checkpoints-kurrent-<homme>`, uus tühi
kaust, ning väljundmudel saaks vale nime. Konkreetne checkpoint:
`--resume=models/checkpoints-kurrent-20260829/checkpoint-1500`.

Enne jätkamist peavad teenused olema jälle maas (vt „Enne käivitamist").

Lossi võrdluspunktid eelmisest jooksust (20260602) – kui number on
kordades suurem, on midagi valesti:

| samm | loss |
|---|---|
| 10 | 1,5 |
| 100 | 0,60 |
| 1589 (epohh 1 lõpp, 20260602 skaalas) | 0,12 |
| 3178 (epohh 2 lõpp, 20260602 skaalas) | 0,08 |

NB: 20260602 jooksus oli epohh 1589 sammu (12 712 lk), nüüd 2122 (16 971 lk) –
sammunumbrid ei ole otse võrreldavad, lossi tase samal epohhi osal on.

### Pärast treeningut

```bash
venv/bin/python scripts/eval_kurrent.py models/qwen3.5-ocr-kurrent-YYYYMMDD
```

73 holdout-lehte, batch 4, **23,9 min** (19,6 s/lk, 183 lehte/tunnis;
mõõdetud 27.08.2026 voolupiiril 450 W). Võrdle vana mudeli baseline'iga
(26.08.2026, `data/kurrent/eval/qwen3.5-ocr-kurrent-20260602/results.csv`):

| | CER | mediaan | loope |
|---|---|---|---|
| kõik 73 lk | 13,9 % | 6,6 % | 2 |
| 50 lk, mida vana mudel NÄGI | 9,9 % | 6,4 % | 1 |
| 23 lk, mida ei näinud (aaeb, hanse, dresdner, senats) | 22,6 % | 8,6 % | 1 |

**Vaata neid 23 lehte** – ainult need on aus võrdlus, ülejäänud 50 on vanal
mudelil treeningust meeles. Vaata ka `ratio` veergu: alla 0,7 = pool lehte
jäi transkribeerimata, üle 1,4 = loop. Vana mudel loopis 2 lehel, neist
`16590_senatsp_UAT_047_19_017` (hõre kohalolijate nimekiri) genereeris „S."
4096 tokenini. Senatsprotokollid ja dresdner_1665 on nüüd treeningandmetes,
seega peaks just see paranema – kui ei parane, ei olnud andmete lisamine
lahendus.

CER mõõdab vastavust arhiivikorpuse tavadele, mitte VUTT-i kasulikkust.
Otsust ei tee ainult numbri põhjal – lase paar päris lehte re-OCR-i läbi ja
vaata silmaga. Tühja lehe käitumist saab kontrollida ilma GT-ta: võta
VUTT-ist märgendamata tühi versopool ja vaata, kas tuleb `[tühi lehekülg]`.

### Sama holdout llama.cpp GGUF-i peal

Kui tahad mõõta, kas kitsaskoht on mudel või mootor, käib sama holdout ka
llama.cpp serveri kaudu. Pane server käima eraldi tmux-i aknasse (ocr-service
peatatud):

```bash
~/Dokumendid/LLM/llama.cpp/build/bin/llama-server \
    -m models/gguf/kurrent-20260602-Q8_0.gguf \
    --mmproj models/gguf/mmproj-kurrent-20260602-F16.gguf \
    -ngl 99 -c 65536 -np 4 -cb -fa on --host 127.0.0.1 --port 8080

venv/bin/python scripts/eval_kurrent.py \
    --endpoint http://127.0.0.1:8080 kurrent-20260602-Q8_0
```

**`-c` on kokku kõigi slottide peale**, ehk `-np 4 -c 65536` annab 16384
tokenit slotile. Leht vajab ~4000 visuaaltokenit + kuni 4096 väljundit, seega
`-c 32768` (8192/slot) jääks napiks. GGUF-i tegemine: [[llamacpp-gguf]] mälus,
skript `scripts/merge_lora.py` → `convert_hf_to_gguf.py --no-nextn`.

Mõõdetud 27.08.2026, sama 73 lehte, mõlemad 450 W juures:

| | 73 lk | s/lk | lehte/tunnis | GPU | CER (69 lk, ilma loopideta) |
|---|---|---|---|---|---|
| unsloth bf16, batch 4 | 23,9 min | 19,6 | 183 | 25,2 GB | 8,7 % |
| llama.cpp Q8_0, `-np 4` | 4,2 min | 3,5 | 1031 | 12,7 GB | 9,1 % |

**5,6x kiirem, täpsus sama.** Aga: llama.cpp serveril **ei ole
`LoopStopper`-it** (VUTT #227 custom `StoppingCriteria`), ja loopi läks teine
leht kui unslothil – koond-CER 13,9 % → 15,6 % tuli tervenisti sellest ühest
lehest. Enne kui llama.cpp teenusesse läheb, tuleb kordusloopi tuvastus
kliendipoolele uuesti teha. Iga jooks jätab tingimused faili
`data/kurrent/eval/<nimi>/run.json` – kiirusnumbrit ei tohi mälu järgi
tsiteerida.

### llama.cpp mõlema mudeli all (aktiveeritud 27.08.2026, katseline)

Mõlemad tüübid käivad nüüd llama.cpp kaudu. Iga mudel vajab OMA serverit:

| tüüp | port | mudel |
|---|---|---|
| print | 8080 | `print-base-r64-mi-vl-20260828-Q8_0.gguf` + `mmproj-…-F16.gguf` |
| hand | 8081 | `kurrent-20260602-Q8_0.gguf` + `mmproj-kurrent-20260602-F16.gguf` |

Trükimudel on vahepeal kaks korda vahetunud (`markup-20260722` →
`print-base-r64-20260827` → `print-base-r64-mi-vl-20260828`); ajalugu ja
mõõtmised on `docs/SEIS.md`-s.

Mõlemad mahuvad korraga GPU-le: **22,1 GB / 32,6 GB**.

```bash
sudo cp systemd/llama-server-{print,hand}.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server-print llama-server-hand
curl -s http://127.0.0.1:8080/health && curl -s http://127.0.0.1:8081/health
sudo systemctl restart ocr-service
```

Teenus kontrollib käivitamisel mõlemat serverit ja keeldub startimast, kui üks
ei vasta. Lüliti on `ENGINE_CONFIGS` failis `kataloogi-jalgimine-ja-ocr.py`.

**Mida see maksab – teadlik kompromiss.** Mõõdetud 143 VUTT-i lehel
(`docs/arhiiv/llamacpp-juurdlus-20260827.md`):

- Käsikiri: **pariteet** (CER 8,8 % vs unslothi 8,7 %), **4,2x kiirem**
- Trükk: 135 puhtal lehel **rohkem** marginaale kui vanas teenuses
  (`<m>` 634 → 684), mediaanlahknevus 3,1 % — **aga 3 lehte 143-st (2,1 %)
  kaotavad marginaaliveeru täielikult**, põhjus teadmata

Need 3 lehte on teadlikult sisse võetud, et saada päris kasutuskogemus.
Kui midagi tundub katki, vaata kõigepealt, kas leht on marginaalirohke.

**Tagasi keeramine (kiire):**
```bash
# ENGINE_CONFIGS mõlemad -> "unsloth"
sudo systemctl restart ocr-service
sudo systemctl disable --now llama-server-print llama-server-hand
```

**Kaks teadaolevat lahtist otsa:**
1. ~~Teenus saadab käsikirjamudelile `INSTRUCTION`-i~~ – **tehtud 01.09.2026.**
   Iga mudel saab nüüd oma treeningjuhise (`INSTRUCTIONS` dict teenuses).
2. Reedese treeningu järel tuleb käsikirjamudel uuesti konverteerida ja
   `llama-server-hand.service` tee uuendada:
   ```bash
   venv/bin/python scripts/merge_lora.py models/qwen3.5-ocr-kurrent-<uus>
   venv/bin/python ~/Dokumendid/LLM/llama.cpp/convert_hf_to_gguf.py \
       models/merged/qwen3.5-ocr-kurrent-<uus>-bf16 --outtype bf16 --no-nextn \
       --outfile models/gguf/kurrent-<uus>-BF16.gguf
   venv/bin/python ~/Dokumendid/LLM/llama.cpp/convert_hf_to_gguf.py \
       models/merged/qwen3.5-ocr-kurrent-<uus>-bf16 --mmproj --outtype f16 \
       --outfile models/gguf/mmproj-kurrent-<uus>-F16.gguf
   ~/Dokumendid/LLM/llama.cpp/build/bin/llama-quantize \
       models/gguf/kurrent-<uus>-BF16.gguf models/gguf/kurrent-<uus>-Q8_0.gguf Q8_0 28
   ```

### Aktiveerimine

Käsikirjamudel elab `kataloogi-jalgimine-ja-ocr.py` failis:

```python
MODEL_CONFIGS = {
    "print": "models/qwen3.5-ocr-print-base-r64-mi-vl-20260828",
    "hand":  "models/qwen3.5-ocr-kurrent-20260602",   # <- see rida
}
```

Pärast muutmist `sudo systemctl restart ocr-service`. Tagasi keeramine =
sama rida vana teega, teenus taaskäivitada.

**Juhis käib mudeliga kaasa** (alates 01.09.2026). Teenuses on
`INSTRUCTIONS = {"print": INSTRUCTION, "hand": KURRENT_INSTRUCTION}`; iga mudel
saab selle juhise, millega ta treeniti. Kuni 01.09 sai ka käsikirjamudel
trükijuhise – CER-is oli vahe 0,1 pp (eristamatu), aga väljundivorm erineb
nähtavalt: trükijuhisega tuleb reavahetuse sidekriipsuks `-`, treeningjuhisega
`¬` nagu VUTT-is kokku lepitud. Kui vahetad `MODEL_CONFIGS`-is mudeli teistsuguse
juhisega treenitu vastu, uuenda ka `INSTRUCTIONS`.
Võrdlus kahe juhise vahel: `scripts/eval_kurrent.py --prompt print`.

---

## Mudeli testimine

Enne aktiveerimist testi treenitud mudelit `data/test/` piltidega:

```bash
source venv/bin/activate

# Kõik testpildid, aktiivne mudel
python scripts/test_model.py

# Kõik testpildid, äsja treenitud mudel
python scripts/test_model.py --model models/qwen3.5-ocr-markup-YYYYMMDD

# Üks konkreetne pilt
python scripts/test_model.py --model models/qwen3.5-ocr-markup-YYYYMMDD data/test/pilt.jpg
```

---

## Uue mudeli aktiveerimine

Kui treening on lõppenud ja tulemus rahuldab:

```bash
sudo systemctl stop ocr-service
DATE=$(date +%Y%m%d)
cp -r models/qwen3.5-ocr-lora models/qwen3.5-ocr-lora-backup-$DATE
rm -rf models/qwen3.5-ocr-lora
cp -r models/qwen3.5-ocr-markup-$DATE models/qwen3.5-ocr-lora
sudo systemctl start ocr-service
```

Tagasipööramine, kui midagi läheb valesti:
```bash
sudo systemctl stop ocr-service
rm -rf models/qwen3.5-ocr-lora
cp -r models/qwen3.5-ocr-lora-backup-$DATE models/qwen3.5-ocr-lora
sudo systemctl start ocr-service
```

---

## Andmestikud

| Kataloog | Sisu | Maht |
|---|---|---|
| `data/lehekyljed/` | 1500 lk, Kreeka + ladina, puhas tekst | etapp 1 treening |
| `data/processed/` | 136 lk, käsitsi märgendatud, markup | markup treening |
| `data/vutt/` | VUTT Valmis lehed, markup | markup treening |
| `data/kurrent/` | 18 908 lk käsikirja (v2, 02.10.2026), `scripts/build_kurrent_v2.py` | Kurrent treening |
| `data/kurrent/holdout.txt` | 73 lk, treeningust väljas | mudelite võrdlus |
| `~/vutt-backups/latest/data` | VUTT backup-snapshot (öine cron) | lähteandmed |

## Mudelid

| Kataloog | Sisu |
|---|---|
| `models/qwen3.5-ocr-print-base-r64-mi-vl-20260828/` | **aktiivne trükimudel** (ocr-service, `print`) |
| `models/qwen3.5-ocr-kurrent-20260602/` | **aktiivne käsikirjamudel** (ocr-service, `hand`) |
| `models/qwen3.5-ocr-lora-backup-20260527/` | etapp 1, puhas transkriptsioon – markup-treeningu lähtepunkt |
| `models/qwen3.5-ocr-markup-YYYYMMDD/` | uued trükimudeli checkpointid |
| `models/qwen3.5-ocr-kurrent-YYYYMMDD/` | uued käsikirjamudeli checkpointid |

Aktiivsed teed on `kataloogi-jalgimine-ja-ocr.py` failis `MODEL_CONFIGS`-is;
koopiaid `models/qwen3.5-ocr-lora/` alla enam ei tehta.

---

## Teenuse haldamine

```bash
sudo systemctl status ocr-service
sudo systemctl start ocr-service
sudo systemctl stop ocr-service
journalctl -u ocr-service -f        # logid reaalajas
tail -f ocr-service.log             # skripti oma logi
```

