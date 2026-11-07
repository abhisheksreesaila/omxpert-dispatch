#!/bin/bash
# stop every FreeToken server/worker and the night-6 runner (pattern kept in this file so pkill never matches the caller)
pkill -f "scripts/n6_ft.sh"; pkill -f "n6_bench.py http://127.0.0.1:18091"; pkill -f "ft serve"
sleep 3
for p in $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep engines/ft | cut -d, -f1); do kill $p; done
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
