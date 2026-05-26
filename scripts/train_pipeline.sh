#!/bin/bash
# OCR markup-treeningu pipeline
#
# Käivitamine (qwen3.5/ kataloogist):
#   bash scripts/train_pipeline.sh          # täistreening
#   bash scripts/train_pipeline.sh --test   # testjooks (5 sammu, ei salvesta)
#
# Eeldused:
#   - venv on aktiveeritud: source venv/bin/activate
#   - SSH ligipääs VUTT serverile toimib: ssh vutt

set -e  # katkesta vea korral

TEST_FLAG=""
if [[ "$1" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "*** TESTREŽIIM ***"
fi

echo "=== 1/5  VUTT andmete sünkroniseerimine ==="
python scripts/vutt_sync.py

echo ""
echo "=== 2/5  Valmis lehtede andmestiku ehitamine ==="
python scripts/build_vutt_dataset.py

echo ""
echo "=== 3/5  OCR-teenuse peatamine (vabastab GPU mälu) ==="
sudo systemctl stop ocr-service

echo ""
echo "=== 4/5  Treenimine ==="
python scripts/train_markup.py $TEST_FLAG

echo ""
echo "=== 5/5  OCR-teenuse taaskäivitamine ==="
sudo systemctl start ocr-service

echo ""
echo "Pipeline lõpetatud."
echo "Uus checkpoint: models/qwen3.5-ocr-markup-$(date +%Y%m%d)/"
echo ""
echo "Aktiveeri uus mudel (kui tulemus rahuldab):"
echo "  sudo systemctl stop ocr-service"
DATE=$(date +%Y%m%d)
echo "  cp -r models/qwen3.5-ocr-lora models/qwen3.5-ocr-lora-backup-$DATE"
echo "  rm -rf models/qwen3.5-ocr-lora"
echo "  cp -r models/qwen3.5-ocr-markup-$DATE models/qwen3.5-ocr-lora"
echo "  sudo systemctl start ocr-service"
