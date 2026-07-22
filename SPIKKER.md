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

### 1. VUTT andmete sünkroniseerimine
```bash
python scripts/vutt_sync.py --dry   # vaata mis muutuks
python scripts/vutt_sync.py         # päris sünk
```
Tulemus: `data/vutt-raw/` (1260+ teost)

### 2. Andmestiku ehitamine
```bash
python scripts/build_vutt_dataset.py --stats   # statistika
python scripts/build_vutt_dataset.py           # kirjuta failid
python scripts/build_vutt_dataset.py --type hand    # käsikirjad (vaikimisi print)
```
Tulemus: `data/vutt/metadata.csv` + `data/vutt/images/`

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
| `data/vutt-raw/` | VUTT toortõmmis (rsync) | vahekataloog |

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

