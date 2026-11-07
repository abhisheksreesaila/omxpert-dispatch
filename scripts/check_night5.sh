#!/bin/bash
# Night 5 milestone checker: PASS/FAIL per milestone in GOAL_NIGHT5.md
cd ~/expert-dispatch
R=results/night5
py() { pixi run -q python -c "$1" 2>/dev/null; }
chk() { if [ "$2" = "ok" ]; then echo "PASS $1"; else echo "FAIL $1 ${2}"; fi; }
chk "1 baseline" "$(py "
import json,os; b=json.load(open('$R/baseline.json'))
assert all(k in b for k in ('decode_tok_s','prefill_tok_s','ram_used_gb')) and b['decode_tok_s']>0
assert os.path.getsize('$R/kl_base.bin')>1e6; print('ok')")"
chk "2 exact tricks" "$(py "
import json; e=json.load(open('$R/exact.json'))['tricks']
need={'staged_prefill','exclusive_tiers','cpu_lane_overlap','read_pool','layer_ahead_prefetch','no_host_sync','mtp','engram_ssd'}
assert need<=set(e); assert all(e[k]['status'] in ('measured','not_built') for k in need)
assert all(e[k].get('reason') for k in need if e[k]['status']=='not_built')
assert sum(e[k]['status']=='measured' for k in need)>=3; print('ok')")"
chk "3 learned tricks" "$(py "
import json; e=json.load(open('$R/learned.json'))['tricks']
need={'profile_residency','learned_prefetcher','steering','adapter'}; assert need<=set(e)
m=[k for k in need if e[k]['status']=='measured']; assert len(m)>=3
assert all('kl' in e[k] or e[k].get('exact') for k in m); print('ok')")"
chk "4 runnable stack" "$(py "
import json,os; s=json.load(open('$R/stack.json'))
assert all(k in s for k in ('baseline_tok_s','exact_stack_tok_s','exact_stack_kl')) and s.get('smoke_ok') is True
assert os.access('omxpert/run.sh',os.X_OK); print('ok')")"
chk "5 leverage" "$(py "
import json; l=json.load(open('$R/leverage.json'))
assert abs(l['check']['err'])<=0.30
assert {'second_nvme','ram_128gb','egpu','npu','mac'}<=set(l['whatif']); print('ok')")"
chk "6 report" "$( [ -f docs/report_night5.md ] && grep -q 0xSero docs/report_night5.md && [ -f docs/visuals/night5_before_after.html ] && grep -qE '^### (5[89]|6[0-9])\.' IDEAS.md && echo ok)"
