# OCR mudeli treenimise spikker

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
```
Tulemus: `data/vutt/metadata.csv` + `data/vutt/images/`

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

