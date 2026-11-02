#!/usr/bin/env bash
# Trükimudel 20261006 hindamine: GGUF + vana ja uus korraga GPU-l, järjest testitud.
# Mall: kurrent_v2_eval.sh (sealt ka põhjendused Q8_0 ja portide kohta).
#
#   tmux new -s truki-eval 'bash scripts/truki_v2_eval.sh 2>&1 | tee logs/truki-v2-eval-$(date +%Y%m%d-%H%M).log'
#
# JUHIS ON MUDELIGA SEOTUD. Uus mudel treeniti makronireegliga
# (logs/prompt-makron-20261006.patch), vana ilma. Kõik vana mudeli sammud
# jooksevad HEAD-juhisega, siis rakendatakse patch ja jooksevad uue sammud;
# trap taastab prompt.py HEAD-i ka katkestusel. eval_print.py võrdsustab
# tilde ja makroni, nii et CER on võrreldav.
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5

UUS=qwen3.5-ocr-print-base-r64-mi-vl-20261006
TAG=print-base-r64-mi-vl-20261006
VANA_TAG=print-base-r64-mi-vl-20260828
PATCH=logs/prompt-makron-20261006.patch
LLAMA=~/Dokumendid/LLM/llama.cpp
PY=venv/bin/python

die() { echo "STOPP: $*" >&2; exit 1; }
samm() { echo; echo "=== $(date +%H:%M) $*"; }

# --- 0. eeldused -----------------------------------------------------------
samm "eeldused"
pgrep -f train_markup.py >/dev/null && die "treening käib"
[[ -f models/$UUS/adapter_model.safetensors ]] || die "models/$UUS adapter puudub"
for s in ocr-service llama-server-print llama-server-hand; do
    systemctl is-active --quiet "$s" && die "$s töötab — GPU mälu ei jätku, peata enne"
done
[[ -f models/gguf/$VANA_TAG-Q8_0.gguf ]] || die "vana GGUF puudub"
git diff --quiet HEAD -- scripts/prompt.py || die "scripts/prompt.py erineb HEAD-ist"
git apply --check $PATCH || die "makronipatch ei rakendu"
$PY scripts/eval_print.py --dry-run --keep-m-italics | tail -n 8

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

# --- 2. kaks serverit korraga (tootmisega samad lipud) ----------------------
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
trap 'kill $P1 $P2 2>/dev/null || true; git checkout -- scripts/prompt.py' EXIT
for port in 8091 8092; do
    for _ in $(seq 120); do
        curl -sf http://127.0.0.1:$port/health >/dev/null && break
        sleep 5
    done
    curl -sf http://127.0.0.1:$port/health >/dev/null || die "server :$port ei tõusnud (logs/)"
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# --- 3. vana mudel, HEAD-juhis ---------------------------------------------
# Vana uuesti (mitte 29.08 kaust): GT on vahepeal makroniks teisendatud ja
# täisjooks samades tingimustes kontrollib ühtlasi determinismi.
samm "holdout: vana Q8_0 (HEAD-juhis)"
$PY scripts/eval_print.py --keep-m-italics --endpoint http://127.0.0.1:8091 $VANA_TAG-Q8_0-v2gt
samm "Menii GGUF: vana"
$PY scripts/menii_gguf.py vana-$VANA_TAG=http://127.0.0.1:8091
samm "Toores-lehed: vana"
$PY scripts/reocr_vutt.py --endpoint http://127.0.0.1:8091 $VANA_TAG-Q8_0-20261006 | tail -n 40

# --- 4. uus mudel, makronijuhis --------------------------------------------
samm "makronipatch peale (trap taastab)"
git apply $PATCH
grep -n "macron" scripts/prompt.py
samm "holdout: uus Q8_0"
$PY scripts/eval_print.py --keep-m-italics --endpoint http://127.0.0.1:8092 $TAG-Q8_0
samm "Menii GGUF: uus"
$PY scripts/menii_gguf.py uus-$TAG=http://127.0.0.1:8092
samm "Toores-lehed: uus"
$PY scripts/reocr_vutt.py --endpoint http://127.0.0.1:8092 $TAG-Q8_0 | tail -n 40

# --- 5. determinism (uus, 4 lehte uuesti) ------------------------------------
samm "determinism: uus, 4 holdout-lehte teist korda"
$PY scripts/eval_print.py --keep-m-italics --limit 4 --endpoint http://127.0.0.1:8092 $TAG-Q8_0-det
for f in data/vutt/eval/$TAG-Q8_0-det/*.txt; do
    cmp -s "$f" "data/vutt/eval/$TAG-Q8_0/$(basename "$f")" && echo "sama   $(basename "$f")" \
        || echo "ERINEB $(basename "$f")"
done

samm "valmis — serverid suletakse, prompt.py taastatakse HEAD-i."
echo "Tootmisvahetus: vt logs/truki-v2-tootmisse.md"
