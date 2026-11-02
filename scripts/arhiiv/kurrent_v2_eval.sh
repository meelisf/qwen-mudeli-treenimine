#!/usr/bin/env bash
# Kurrent v2 (20261002) hindamine: GGUF + vana ja uus korraga GPU-l, järjest testitud.
#
#   tmux new -s eval2 'bash scripts/kurrent_v2_eval.sh 2>&1 | tee logs/kurrent-v2-eval-$(date +%Y%m%d-%H%M).log'
#   bash scripts/kurrent_v2_eval.sh --bf16    # + unsloth bf16 holdout (~46 min, GPU)
#
# Miks llama.cpp Q8_0, mitte unsloth bf16:
#   - tootmine (llama-server-hand) jookseb Q8_0 GGUF-iga — see on see number,
#     mis loeb; ja ta on ~5x kiirem (4,5 vs 21 s/lk, 133 lk ≈ 10 min mudeli kohta)
#   - kaks bf16 mudelit (~18 GB + aktivatsioonid kumbki) ei mahu korraga 32,6 GB
#     peale; kaks Q8_0 serverit mahuvad (tootmises 22,1 GB koos print-mudeliga)
# Mõlemad serverid on korraga üleval, aga eval jookseb JÄRJEST: paralleelselt
# jagaksid nad GPU-d ja s/lk numbrid oleksid võrreldamatud.
#
# Pordid 8091/8092, mitte 8080/8081 — kui keegi teenuse vahepeal käivitab,
# ei lähe tema liiklus siia ega vastupidi.
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5

UUS=qwen3.5-ocr-kurrent-20261002
TAG=kurrent-20261002
VANA_TAG=kurrent-20260829
LLAMA=~/Dokumendid/LLM/llama.cpp
PY=venv/bin/python
BF16=0
[[ "${1:-}" == "--bf16" ]] && BF16=1

die() { echo "STOPP: $*" >&2; exit 1; }
samm() { echo; echo "=== $(date +%H:%M) $*"; }

# --- 0. eeldused -----------------------------------------------------------
samm "eeldused"
pgrep -f train_kurrent.py >/dev/null && die "treening käib veel"
[[ -f models/$UUS/adapter_model.safetensors ]] || die "models/$UUS adapter puudub"
for s in ocr-service llama-server-print llama-server-hand; do
    systemctl is-active --quiet "$s" && die "$s töötab — GPU mälu ei jätku, peata enne"
done
[[ -f models/gguf/$VANA_TAG-Q8_0.gguf ]] || die "vana GGUF puudub"
[[ -f data/kurrent/oesel/m3do2t_lk003.jpeg ]] || die "Oeseli pilt puudub"
$PY scripts/eval_kurrent.py --dry-run | tail -n 3

# --- 1. GGUF (CPU) ± bf16 holdout (GPU) paralleelselt ---------------------
BF16_PID=""
if (( BF16 )); then
    samm "unsloth bf16 holdout taustal (GPU)"
    $PY scripts/eval_kurrent.py models/$UUS > logs/$TAG-bf16-eval.log 2>&1 &
    BF16_PID=$!
fi

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

if [[ -n "$BF16_PID" ]]; then
    samm "ootan bf16 holdouti lõppu (vt logs/$TAG-bf16-eval.log)"
    wait "$BF16_PID" || die "bf16 eval kukkus, vt logs/$TAG-bf16-eval.log"
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
trap 'kill $P1 $P2 2>/dev/null || true' EXIT
for port in 8091 8092; do
    for _ in $(seq 120); do
        curl -sf http://127.0.0.1:$port/health >/dev/null && break
        sleep 5
    done
    curl -sf http://127.0.0.1:$port/health >/dev/null || die "server :$port ei tõusnud (logs/)"
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# --- 3. holdout 133, järjest ----------------------------------------------
# Vana läheb UUDE kausta (-h133): 73 lehte on vanas kaustas olemas, aga
# täisjooks samades tingimustes on 10 min ja kontrollib ühtlasi determinismi.
samm "holdout: vana Q8_0"
$PY scripts/eval_kurrent.py --endpoint http://127.0.0.1:8091 $VANA_TAG-Q8_0-h133
samm "holdout: uus Q8_0"
$PY scripts/eval_kurrent.py --endpoint http://127.0.0.1:8092 $TAG-Q8_0

# --- 4. Oeseli leht -------------------------------------------------------
samm "Oesel m3do2t lk 3"
$PY scripts/oesel_proov.py vana=http://127.0.0.1:8091 uus=http://127.0.0.1:8092

# --- 5. võrdlus -----------------------------------------------------------
samm "võrdlus"
JOOKSUD=($VANA_TAG-Q8_0-h133 $TAG-Q8_0)
(( BF16 )) && JOOKSUD+=(qwen3.5-ocr-kurrent-20260829 $UUS)
$PY scripts/vordle_kurrent.py "${JOOKSUD[@]}"
# Determinism: vana 29.08 kaust (73 lk) vs täna sama mudel
# (peaks olema „parem 0, halvem 0")
$PY scripts/vordle_kurrent.py $VANA_TAG-Q8_0 $VANA_TAG-Q8_0-h133 | grep " vs " 

samm "valmis — serverid suletakse. OCR-teenused käivita ise:"
echo "  sudo systemctl start llama-server-print llama-server-hand && sudo systemctl restart ocr-service"
echo "  (llama-server-hand osutab endiselt $VANA_TAG-ile — vahetus alles pärast otsust)"
