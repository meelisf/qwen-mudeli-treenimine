#!/bin/bash
# Öine järeltöö 30./31.08.2026: oota Kurrendi treeningu lõppu, tee GGUF,
# hinda holdout'il nii GGUF-i kui transformersi peal.
# EI aktiveeri midagi tootmises, EI puutu systemd-teenuseid.
cd /home/mf/Dokumendid/LLM/qwen3.5 || exit 1

PID=3121315
STAMP=20260829
MODEL=models/qwen3.5-ocr-kurrent-$STAMP
CKPT=models/checkpoints-kurrent-$STAMP
EP1=models/kurrent-$STAMP-epohh1
MERGED=models/merged/qwen3.5-ocr-kurrent-$STAMP-bf16
GBF16=models/gguf/kurrent-$STAMP-BF16.gguf
GQ8=models/gguf/kurrent-$STAMP-Q8_0.gguf
GMM=models/gguf/mmproj-kurrent-$STAMP-F16.gguf
LC=/home/mf/Dokumendid/LLM/llama.cpp
SP=/tmp/claude-1000/-home-mf-Dokumendid-LLM-qwen3-5/a0c7f425-daa4-463b-99a3-51c9b070a217/scratchpad
PORT=8099
SRVPID=""

say() { echo "[$(date '+%d.%m %H:%M:%S')] MARKER $*"; }
cleanup() { [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; }
trap cleanup EXIT

say "OOTAN treeningu lõppu (pid $PID)"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
say "TREENING-LOPPES"
sleep 120   # lase GPU mälul vabaneda

if [ ! -d "$MODEL" ]; then
    say "VIGA lõppmudelit $MODEL EI OLE - treening kukkus enne salvestamist"
    say "KOIK-VALMIS ebaonnestus"; exit 1
fi
say "MUDEL-OLEMAS $MODEL"

# --- 1. GGUF-ahel ------------------------------------------------------------
say "GGUF-ALGAB merge_lora"
venv/bin/python scripts/merge_lora.py "$MODEL" > "$SP/gguf.log" 2>&1 \
  && venv/bin/python "$LC/convert_hf_to_gguf.py" "$MERGED" --outtype bf16 --no-nextn \
       --outfile "$GBF16" >> "$SP/gguf.log" 2>&1 \
  && venv/bin/python "$LC/convert_hf_to_gguf.py" "$MERGED" --mmproj --outtype f16 \
       --outfile "$GMM" >> "$SP/gguf.log" 2>&1 \
  && "$LC/build/bin/llama-quantize" "$GBF16" "$GQ8" Q8_0 28 >> "$SP/gguf.log" 2>&1
GGUF_OK=$?
say "GGUF-VALMIS exit=$GGUF_OK $(ls -la $GQ8 2>/dev/null | awk '{print $5}') baiti"

# --- 2. hindamine GGUF Q8_0 peal (tootmismootor) -----------------------------
if [ "$GGUF_OK" = "0" ] && [ -f "$GQ8" ]; then
    say "SERVER-ALGAB port $PORT"
    "$LC/build/bin/llama-server" -m "$GQ8" --mmproj "$GMM" \
        --image-max-tokens 5000 -ngl 99 -c 65536 -np 4 -cb -fa on \
        --host 127.0.0.1 --port $PORT > "$SP/server.log" 2>&1 &
    SRVPID=$!
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 10
    done
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        say "EVAL-GGUF-ALGAB"
        venv/bin/python scripts/eval_kurrent.py --endpoint "http://127.0.0.1:$PORT" \
            "kurrent-$STAMP-Q8_0" > "$SP/eval_gguf.log" 2>&1
        say "EVAL-GGUF-VALMIS exit=$?"
    else
        say "VIGA server ei vastanud 10 min jooksul"
    fi
    kill $SRVPID 2>/dev/null; SRVPID=""; sleep 30
else
    say "EVAL-GGUF-VAHELE konversioon kukkus"
fi

# --- 3. hindamine transformers bf16 (õun-õuna vs 20260602 baseline) ----------
say "EVAL-BF16-ALGAB"
venv/bin/python scripts/eval_kurrent.py "$MODEL" > "$SP/eval_bf16.log" 2>&1
say "EVAL-BF16-VALMIS exit=$?"

# --- 4. 1. epohhi adapter: kas 2. epohh andis midagi -------------------------
if [ -d "$CKPT/epohh-1-adapter" ]; then
    rm -rf "$EP1"; mkdir -p "$EP1"
    cp "$CKPT/epohh-1-adapter/adapter_config.json" \
       "$CKPT/epohh-1-adapter/adapter_model.safetensors" "$EP1/"
    for f in tokenizer.json tokenizer_config.json chat_template.jinja \
             processor_config.json preprocessor_config.json \
             special_tokens_map.json added_tokens.json video_preprocessor_config.json; do
        [ -f "$MODEL/$f" ] && cp "$MODEL/$f" "$EP1/"
    done
    say "EVAL-EPOHH1-ALGAB"
    venv/bin/python scripts/eval_kurrent.py "$EP1" --name kurrent-$STAMP-epohh1 \
        > "$SP/eval_ep1.log" 2>&1
    say "EVAL-EPOHH1-VALMIS exit=$?"
else
    say "EVAL-EPOHH1-VAHELE adapter puudub"
fi

say "KOIK-VALMIS"
