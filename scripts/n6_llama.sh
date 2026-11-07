#!/bin/bash
# Night 6: start our llama.cpp stack (omxpert/run.sh: -lzm off, MTP=1 for the draft head) and run the shared client.
# usage: [MTP=1] [STACK_ARGS=...] n6_llama.sh <name>      port 18081
cd ~/expert-dispatch; name=$1; export PORT=18081 CTX=${CTX:-4096}
omxpert/run.sh > results/night6/llama_$name.log 2>&1 &
PID=$!
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health | grep -q ok; do
  kill -0 $PID 2>/dev/null || { echo "llama died"; tail -20 results/night6/llama_$name.log; exit 1; }
  sleep 3
done
echo "loaded in $(( $(date +%s) - t0 )) s"
f0=$(awk '{print $12}' /proc/$PID/stat)
pixi run -q python scripts/n6_bench.py http://127.0.0.1:$PORT $name results/night6/$name.json; rc=$?
f1=$(awk '{print $12}' /proc/$PID/stat)
pixi run -q python -c "
import json,sys; p='results/night6/$name.json'; d=json.load(open(p)); n=sum(r['completion_tokens']+(r['prompt_tokens'] or 0) for r in d['runs'])
d['majflt']=$f1-$f0; d['majflt_per_token']=round(($f1-$f0)/max(n,1),1); d['stack_args']='$STACK_ARGS'; d['stack_env']='$STACK_ENV'
d['pf_stats']=open('/tmp/n6pf.ctl.stats').read() if __import__('os').path.exists('/tmp/n6pf.ctl.stats') else None
json.dump(d,open(p,'w'),indent=1); print('majflt/token',d['majflt_per_token'])"
kill $PID; wait $PID 2>/dev/null; exit $rc
