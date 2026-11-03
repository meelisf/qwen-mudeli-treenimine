#!/usr/bin/env bash
# Trükimudel print-base-r64-mi-vl-20261102 tootmisse (+ absoluutne <f>-juhis, VUTT ADR 0077).
# Käivita LOSSis ise (sudo küsib parooli), PÄRAST seda, kui hindamine on üle vaadatud:
#
#   bash scripts/truki_v3_tootmisse.sh
#
# Mudel ja juhis käivad KOOS: ocr-service loeb juhise stardil. Skript rakendab
# logs/truki-v3-tootmisse.patch (unit + "print": INSTRUCTION) ja commitib selle,
# paigaldab unit'i ja alles SIIS käivitab teenused. VUTT-is peab samal ajal minema
# tootmisse config.OCR_OUTPUT_RELATIVE_CS = False (VUTT PR) — muidu teisendab VUTT
# uue absoluutse <f>-väljundi uuesti cs_to_f-iga ja pöörab fraktuurlehed tagurpidi.
#
# TAGASI (vana mudel + vana <cs>-juhis):
#   sudo systemctl stop ocr-service llama-server-print
#   sudo cp /etc/systemd/system/llama-server-print.service.bak-20261102 /etc/systemd/system/llama-server-print.service
#   git revert <selle vahetuse commit>     # unit + "print": INSTRUCTION_CS tagasi
#   sudo systemctl daemon-reload && sudo systemctl start llama-server-print ocr-service
#   (+ VUTT-is OCR_OUTPUT_RELATIVE_CS = True tagasi)
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5
UNIT=/etc/systemd/system/llama-server-print.service
TAG=print-base-r64-mi-vl-20261102
PATCH=logs/truki-v3-tootmisse.patch

die() { echo "STOPP: $*" >&2; exit 1; }
[[ -f models/gguf/$TAG-Q8_0.gguf && -f models/gguf/mmproj-$TAG-F16.gguf ]] || die "GGUF puudub"
pgrep -f "llama-server.*--port 809[12]" >/dev/null && die "hindamise server veel üleval (tmux truki-eval)"
git diff --quiet HEAD -- kataloogi-jalgimine-ja-ocr.py systemd/llama-server-print.service \
    || die "kataloogi-jalgimine-ja-ocr.py või unit erineb HEAD-ist"

if grep -q "$TAG-Q8_0" systemd/llama-server-print.service; then
    echo "patch juba sees"
else
    git apply --check -p1 $PATCH || die "patch ei rakendu"
    git apply -p1 $PATCH
    venv/bin/python -c "import ast,sys; ast.parse(open('kataloogi-jalgimine-ja-ocr.py').read())"
    git add kataloogi-jalgimine-ja-ocr.py systemd/llama-server-print.service
    git commit -m "Trükimudel $TAG tootmisse + absoluutne <f>-juhis (VUTT ADR 0077)"
fi
grep -q '"print": INSTRUCTION,' kataloogi-jalgimine-ja-ocr.py || die "juhis pole INSTRUCTION"

sudo systemctl stop ocr-service || true          # vana juhis ei tohi uue mudeliga jooksta
[[ -f $UNIT.bak-20261102 ]] || sudo cp $UNIT $UNIT.bak-20261102
sudo cp systemd/llama-server-print.service $UNIT
sudo systemctl daemon-reload
sudo systemctl restart llama-server-print
sudo systemctl start llama-server-hand

for port in 8080 8081; do
    for _ in $(seq 60); do curl -sf http://127.0.0.1:$port/health >/dev/null && break; sleep 5; done
    curl -sf http://127.0.0.1:$port/health >/dev/null || die "llama :$port ei tõusnud (journalctl -u llama-server-*)"
done
sudo systemctl start ocr-service
sleep 5
systemctl is-active ocr-service llama-server-print llama-server-hand
ps -o args= -C llama-server | grep -o "gguf/[^ ]*Q8_0.gguf"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "OK. Logi: tail -f ocr-service.log. Ära unusta: git push + VUTT PR deploy."
