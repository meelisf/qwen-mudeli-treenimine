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
`docs/katkised-lehed-20260721.txt`.

### ⚠️ Tühjad leheküljed puuduvad treeningandmetest (lahendamata)

**Probleem:** kui mudelile anda tühi (või peaaegu tühi) lehekülg, läheb ta
loopi ja genereerib maksimumini täiesti suvalist teksti. Tagajärg on
kahtlaselt vastupidine intuitsioonile: **tühi lehekülg võtab praegu
tunduvalt rohkem aega kui tekstiga lehekülg** — tekstiga leht lõpetab
loomulikult EOS-iga, tühi leht jookseb iga kord token-lakke (~8 min lehe
kohta). Kui partiis on tühje lehti, on aeglus just nende taga, mitte
raske teksti taga.

Põhjus on lihtne: **mudel ei ole kunagi näinud ühtegi näidet, kus õige
vastus on „lehekülg on tühi"**, seega pole tal midagi, mille peal lõpetada.

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
- **Vahepealsed juhud** (leheküljel ainult signatuur, ainult leheküljenumber,
  ainult tint-plekk) transkribeeritakse tavaliselt – tühja lehe märgend
  ainult tõesti tühjale.
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
  ainus tõeallikas ja tühja lehe reegel on lisatud mõlemasse juhisesse –
  `INSTRUCTION` (trükk/markup) ja `KURRENT_INSTRUCTION` (käsikiri).
- `build_vutt_dataset.py`: uued lipud `--out`, `--append`, `--allikas`,
  `--only-empty`, `--force` + tühja lehe märgendi kontroll.

**Käsikirjade tühjade lehtede lisamine 17 000 lk andmestikule:**

```bash
# 1. Vaata üle, mis VUTT-ist tuleb (ei kirjuta midagi)
python scripts/build_vutt_dataset.py --type hand --only-empty --stats

# 2. Lisa Kurrent-andmestikule
python scripts/build_vutt_dataset.py --type hand --only-empty \
    --out data/kurrent --append --allikas vutt_tyhjad
```

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
| `~/vutt-backups/latest/data` | VUTT backup-snapshot (öine cron) | lähteandmed |

## Mudelid

| Kataloog | Sisu |
|---|---|
| `models/qwen3.5-ocr-lora/` | **aktiivne mudel** (ocr-service kasutab) |
| `models/qwen3.5-ocr-lora-stage2/` | vanem markup mudel (136 lk) |
| `models/qwen3.5-ocr-markup-YYYYMMDD/` | uued treenitud checkpointid |

---

## Teenuse haldamine

```bash
sudo systemctl status ocr-service
sudo systemctl start ocr-service
sudo systemctl stop ocr-service
journalctl -u ocr-service -f        # logid reaalajas
tail -f ocr-service.log             # skripti oma logi
```

