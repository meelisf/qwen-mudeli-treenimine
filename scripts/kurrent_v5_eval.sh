#!/usr/bin/env bash
# Kurrent v5 (20261013, TÜ HPC) hindamine: v4 (20261009) vs v5 Q8_0, v5m holdout.
#
#   tmux new -s eval5 "bash scripts/kurrent_v5_eval.sh 2>&1 | tee logs/kurrent-v5-eval-$(date +%Y%m%d-%H%M).log"
#
# Erinevused v4 skriptist (kurrent_v4_eval.sh — miks Q8_0, pordid jne vt sealt):
#   - GT = data/kurrent_v5m (--data-dir), mitte vana data/kurrent: 154 lehte, sh
#     uued Leibniz/DTA lehed ja 8 <m>-lehte. Vana holdout on selle sees.
#   - Kumbki mudel saab OMA juhise: v4 --prompt kurrent, v5 --prompt kurrent_v5.
#     Muidu ei ole võrdlus aus (vt mälu project_kurrent_v5_juhis).
#   - cer/wer on <m>/<add> märgenditeta (v4 neid ei kirjuta); märgendid
#     loetakse eraldi veergudes gt_m/out_m/gt_add/out_add, cer_raw märgenditega.
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5

UUS=qwen3.5-ocr-kurrent-20261013
TAG=kurrent-20261013
VANA_TAG=kurrent-20261009
DATA=data/kurrent_v5m
LLAMA=~/Dokumendid/LLM/llama.cpp
PY=venv/bin/python

die() { echo "STOPP: $*" >&2; exit 1; }
samm() { echo; echo "=== $(date +%H:%M) $*"; }

# --- 0. eeldused -----------------------------------------------------------
samm "eeldused"
pgrep -f train_kurrent.py >/dev/null && die "treening käib"
[[ -f models/$UUS/adapter_model.safetensors ]] || die "models/$UUS adapter puudub"
for s in ocr-service llama-server-print llama-server-hand; do
    systemctl is-active --quiet "$s" && die "$s töötab — GPU mälu ei jätku, peata enne"
done
[[ -f models/gguf/$VANA_TAG-Q8_0.gguf ]] || die "vana GGUF puudub"
[[ -f data/kurrent/oesel/m3do2t_lk003.jpeg ]] || die "Oeseli pilt puudub"
$PY scripts/eval_kurrent.py --data-dir=$DATA --dry-run | tail -n 3

# --- 1. GGUF (CPU) ---------------------------------------------------------
if [[ -f models/gguf/$TAG-Q8_0.gguf && -f models/gguf/mmproj-$TAG-F16.gguf ]]; then
    samm "GGUF juba olemas, jätan vahele"
else
    samm "merge (CPU, ~18 GB RAM)"
    [[ -d models/merged/$UUS-bf16 ]] || $PY scripts/merge_lora.py models/$UUS
    samm "convert bf16 + mmproj"
    $PY $LLAMA/convert_hf_to_gguf.py models/merged/$UUS-bf16 --outtype bf16 --no-nextn \
        --outfile models/gguf/$TAG-BF16.gguf
    $PY $LLAMA/convert_hf_to_gguf.py models/merged/$UUS-bf16 --mmproj --outtype f16 \
        --outfile models/gguf/mmproj-$TAG-F16.gguf
    samm "quantize Q8_0"
    $LLAMA/build/bin/llama-quantize models/gguf/$TAG-BF16.gguf models/gguf/$TAG-Q8_0.gguf Q8_0 28
fi

# --- 2. kaks serverit korraga ---------------------------------------------
kaivita() {   # $1 = tag, $2 = port
    $LLAMA/build/bin/llama-server \
        -m models/gguf/$1-Q8_0.gguf --mmproj models/gguf/mmproj-$1-F16.gguf \
        --image-max-tokens 5000 \
        -ngl 99 -c 65536 -np 4 -cb -fa on --host 127.0.0.1 --port "$2" \
        > "logs/llama-$1-$2.log" 2>&1 &
    echo $!
}
samm "llama-serverid: vana :8091, uus :8092"
P1=$(kaivita $VANA_TAG 8091)
P2=$(kaivita $TAG 8092)
trap "kill $P1 $P2 2>/dev/null || true" EXIT
for port in 8091 8092; do
    for _ in $(seq 120); do
        curl -sf http://127.0.0.1:$port/health >/dev/null && break
        sleep 5
    done
    curl -sf http://127.0.0.1:$port/health >/dev/null || die "server :$port ei tõusnud (logs/)"
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# --- 3. holdout, järjest --------------------------------------------------
samm "holdout: vana Q8_0 (--prompt kurrent)"
$PY scripts/eval_kurrent.py --data-dir=$DATA --prompt kurrent \
    --endpoint http://127.0.0.1:8091 $VANA_TAG-Q8_0
samm "holdout: uus Q8_0 (--prompt kurrent_v5)"
$PY scripts/eval_kurrent.py --data-dir=$DATA --prompt kurrent_v5 \
    --endpoint http://127.0.0.1:8092 $TAG-Q8_0

# --- 4. Oeseli leht -------------------------------------------------------
samm "Oesel m3do2t lk 3"
$PY scripts/oesel_proov.py vana=http://127.0.0.1:8091 uus=http://127.0.0.1:8092@kurrent_v5

# --- 5. võrdlus -----------------------------------------------------------
samm "võrdlus (cer märgenditeta)"
$PY scripts/vordle_kurrent.py --data-dir=$DATA $VANA_TAG-Q8_0 $TAG-Q8_0

samm "märgendid (<m>, <add>) holdoutis"
$PY - "$DATA/eval" $VANA_TAG-Q8_0 $TAG-Q8_0 <<"PYEOF"
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
for jooks in sys.argv[2:]:
    with open(root / jooks / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s = lambda k: sum(int(r[k]) for r in rows)
    m_lehed = [r for r in rows if int(r['gt_m'])]
    print(f"{jooks}: <m> GT {s('gt_m')} / väljund {s('out_m')}"
          f" ({len(m_lehed)} <m>-lehte, neil väljundis {sum(int(r['out_m']) for r in m_lehed)});"
          f"  <add> GT {s('gt_add')} / väljund {s('out_add')}")
PYEOF

samm "valmis — serverid suletakse. OCR-teenused käivita ise:"
echo "  sudo systemctl start llama-server-print llama-server-hand && sudo systemctl restart ocr-service"
echo "  (llama-server-hand osutab endiselt $VANA_TAG-ile — vahetus alles pärast otsust, koos KURRENT_INSTRUCTION_V5-ga)"
