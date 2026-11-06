#!/bin/bash
# PASS/FAIL per night-3 milestone (see GOAL_NIGHT3.md)
cd ~/expert-dispatch; R=results/OLMoE-1B-7B-0125-Instruct
ok() { [ "$2" = 0 ] && echo "PASS $1" || echo "FAIL $1"; }
[ -f results/night3_linux.done ] && [ -f $R/swapsearch_S8.json ] && [ -f $R/swapsearch_S16.json ]; ok "1 linux queue + learned sets" $?
python3 - <<'PY'; ok "2 learned set + adapter beats counted set + adapter (S=8)" $?
import json,sys,pathlib
R=pathlib.Path.home()/"expert-dispatch/results/OLMoE-1B-7B-0125-Instruct"
try:
    a=json.load(open(R/"adapter_coding_search_S8_linf_r32_t240000.json")); b=json.load(open(R/"adapter_S16_linf_r32_t240000.json".replace("S16","S8")))
except Exception: sys.exit(1)
sys.exit(0 if a["coding_steer_adapter"] < 6.634 else 1)
PY
[ -f results/mac/night3.log ] && grep -q "night3 done" results/mac/night3.log; ok "3 mac sweep collected" $?
curl -s https://abhisheksreesaila.github.io/omxpert-dispatch/ | grep -q "Learned by search"; ok "4 site updated and live" $?
[ -s docs/report_night3.md ]; ok "5 morning report" $?
