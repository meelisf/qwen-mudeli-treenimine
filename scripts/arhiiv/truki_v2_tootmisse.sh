#!/usr/bin/env bash
# Trükimudel print-base-r64-mi-vl-20261006 tootmisse (+ makronijuhis, VUTT ADR 0062).
# Käivita LOSSis ise (sudo küsib parooli):
#
#   bash scripts/truki_v2_tootmisse.sh
#
# Juhis ja mudel käivad KOOS: prompt.py makronirida on juba commititud ja
# ocr-service loeb selle stardil. Seepärast EI TOHI ocr-service'i käivitada
# vana print-mudeliga — skript paigaldab unit'i enne mis tahes starti.
#
# TAGASI (vana mudel + vana juhis):
#   sudo cp /etc/systemd/system/llama-server-print.service.bak-20261006 /etc/systemd/system/llama-server-print.service
#   git revert <selle vahetuse commit>     # prompt.py makronirida välja
#   sudo systemctl daemon-reload && sudo systemctl restart llama-server-print ocr-service
set -euo pipefail
cd ~/Dokumendid/LLM/qwen3.5
UNIT=/etc/systemd/system/llama-server-print.service

die() { echo "STOPP: $*" >&2; exit 1; }
grep -q "U+0304" scripts/prompt.py || die "prompt.py-s pole makronireeglit"
grep -q "print-base-r64-mi-vl-20261006-Q8_0" systemd/llama-server-print.service || die "repo unit vale"
[[ -f models/gguf/print-base-r64-mi-vl-20261006-Q8_0.gguf ]] || die "GGUF puudub"
pgrep -f "llama-server.*--port 809[12]" >/dev/null && die "hindamise serverid veel üleval"

[[ -f $UNIT.bak-20261006 ]] || sudo cp $UNIT $UNIT.bak-20261006
sudo cp systemd/llama-server-print.service $UNIT
sudo systemctl daemon-reload
sudo systemctl start llama-server-print llama-server-hand

for port in 8080 8081; do
    for _ in $(seq 60); do curl -sf http://127.0.0.1:$port/health >/dev/null && break; sleep 5; done
    curl -sf http://127.0.0.1:$port/health >/dev/null || die "llama :$port ei tõusnud (journalctl -u llama-server-*)"
done
sudo systemctl start ocr-service
sleep 5
systemctl is-active ocr-service llama-server-print llama-server-hand
ps -o args= -C llama-server | grep -o "gguf/[^ ]*Q8_0.gguf"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "OK. Logi: tail -f ocr-service.log"
