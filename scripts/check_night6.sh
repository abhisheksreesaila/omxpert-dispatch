#!/bin/bash
# Night 6 milestone checker: PASS/FAIL per milestone in GOAL_NIGHT6.md
cd ~/expert-dispatch
R=results/night6
py() { pixi run -q python -c "$1" 2>/dev/null; }
chk() { if [ "$2" = "ok" ]; then echo "PASS $1"; else echo "FAIL $1 ${2}"; fi; }
chk "1 baseline" "$(py "
import json; b=json.load(open('$R/baseline.json'))
assert b['decode_tok_s']>0 and b['prefill_tok_s']>0 and len(b['runs'])>=2; print('ok')")"
chk "2 freetoken runs" "$(py "
import json; b=json.load(open('$R/ft_stock.json'))
assert b['engine'].startswith('freetoken') and b['smoke_ok'] is True and b['decode_tok_s']>0 and b['prefill_tok_s']>0; print('ok')")"
chk "3 chat profile" "$(py "
import json; p=json.load(open('$R/profile_chat.json'))
assert p['source']=='owner_chat_history' and p['corpus_tokens']>=20000 and len(p['hit_curve'])>=3; print('ok')")"
chk "4 combined" "$(py "
import json; e=json.load(open('$R/combined.json'))['tricks']
need={'freetoken_hybrid','profile_residency','predictor_prefetch','mtp','read_pool','layer_ahead_prefetch','staged_prefill','exclusive_tiers','steering'}
assert need<=set(e) and all(e[k]['status'] in ('measured','not_built') for k in need)
assert all(e[k].get('reason') for k in need if e[k]['status']=='not_built')
m=[k for k in need if e[k]['status']=='measured']; assert len(m)>=3 and all('paired' in e[k] or 'vs' in e[k] for k in m); print('ok')")"
chk "5 quality" "$(py "
import json; q=json.load(open('$R/quality.json'))
assert ('kl' in q or 'token_match' in q) and 'floor' in q and 'verdict' in q; print('ok')")"
chk "6 run + report" "$(py "
import json,os; s=json.load(open('$R/stack.json'))
assert s.get('smoke_ok') is True and s.get('decode_tok_s',0)>0 and os.access('omxpert/run.sh',os.X_OK)
r=open('docs/report_night6.md').read(); assert '0xSero' in r and 'FreeToken' in r
assert os.path.exists('docs/visuals/night6_before_after.html'); print('ok')")"
