#!/usr/bin/env bash
# Trükimudel 20261102 (<f>, ADR 0077) hindamine: GGUF + uus mudel :8092, holdout + determinism.
# Mall: arhiiv/truki_v2_eval.sh. Vana mudelit uuesti ei jooksutata: oktoobri väljundid on
# hinnatud praeguse GT vastu → data/vutt/eval/print-base-r64-mi-vl-20261006-Q8_0-gt0077/.
#
#   tmux new -s truki-eval 'bash scripts/truki_v3_eval.sh 2>&1 | tee logs/truki-v3-eval-$(date +%Y%m%d-%H%M).log'
#
# Juhis: INSTRUCTION (absoluutne <f>), millega uus mudel treeniti. reocr_vutt kirjutab
# ainult kohalikku data/vutt/reocr/ kausta, VUTT-i ei puutu.
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5

UUS=qwen3.5-ocr-print-base-r64-mi-vl-20261102
TAG=print-base-r64-mi-vl-20261102
LLAMA=~/Dokumendid/LLM/llama.cpp
PY=venv/bin/python

die() { echo "STOPP: $*" >&2; exit 1; }
samm() { echo; echo "=== $(date +%H:%M) $*"; }

samm "eeldused"
pgrep -f train_markup.py >/dev/null && die "treening käib"
[[ -f models/$UUS/adapter_model.safetensors ]] || die "models/$UUS adapter puudub"
for s in ocr-service llama-server-print llama-server-hand; do
    systemctl is-active --quiet "$s" && die "$s töötab — GPU mälu ei jätku, peata enne"
done
grep -q "<cs>" <(sed -n '/^INSTRUCTION = """/,/^"""/p' scripts/prompt.py) && die "INSTRUCTION sisaldab <cs>"
$PY scripts/eval_print.py --dry-run --keep-m-italics | tail -n 8

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

samm "llama-server: uus :8092 (tootmisega samad lipud)"
$LLAMA/build/bin/llama-server \
    -m models/gguf/$TAG-Q8_0.gguf --mmproj models/gguf/mmproj-$TAG-F16.gguf \
    --image-max-tokens 5000 \
    -ngl 99 -c 65536 -np 4 -cb -fa on --host 127.0.0.1 --port 8092 \
    > "logs/llama-$TAG-8092.log" 2>&1 &
P=$!
trap 'kill $P 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf http://127.0.0.1:8092/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8092/health >/dev/null || die "server ei tõusnud (logs/)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

samm "holdout: uus Q8_0"
$PY scripts/eval_print.py --keep-m-italics --endpoint http://127.0.0.1:8092 $TAG-Q8_0

samm "determinism: 4 holdout-lehte teist korda"
$PY scripts/eval_print.py --keep-m-italics --limit 4 --endpoint http://127.0.0.1:8092 $TAG-Q8_0-det
for f in data/vutt/eval/$TAG-Q8_0-det/*.txt; do
    cmp -s "$f" "data/vutt/eval/$TAG-Q8_0/$(basename "$f")" && echo "sama   $(basename "$f")" \
        || echo "ERINEB $(basename "$f")"
done

samm "Toores-lehed: uus (40 lehte, ainult kohalik võrdlus)"
$PY scripts/reocr_vutt.py --endpoint http://127.0.0.1:8092 --limit 40 $TAG-Q8_0 | tail -n 40 || true

samm "valmis — server suletakse."
