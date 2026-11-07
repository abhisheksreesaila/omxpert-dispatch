#!/bin/bash
# Night 6: start FreeToken-Kai on Qwen3.8-Flash-Next (our UD-Q4_K_XL GGUF) on 2x3090 + 62 GB RAM, then run the shared client.
# usage: FTARGS="<extra ft serve flags>" n6_ft.sh <name> [bench|serve]     port 18091
cd ~/expert-dispatch
name=$1; mode=${2:-bench}; PORT=${PORT:-18091}
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
LOG=results/night6/ft_$name.log
BASE="--model $GG --gpu 0,1 --pp-size 2 --host 127.0.0.1 --port $PORT --moe-strategy hybrid --max-running-req 1 \
 --max-seq-len-override 8192 --kv-reserve-tokens 8192 --moe-cpu-threads 12"
cmd="nice -n 10 pixi run --manifest-path engines/ft/pixi.toml ft serve $BASE ${FTBASE-} ${FTARGS:-}"
echo "$cmd" > $LOG
if [ "$mode" = serve ]; then exec $cmd >> $LOG 2>&1; fi
$cmd >> $LOG 2>&1 &
PID=$!
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health | grep -q '"serving"'; do
  kill -0 $PID 2>/dev/null || { echo "ft died, see $LOG"; tail -20 $LOG; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt ${LOAD_TIMEOUT:-3600} ] && { echo "load timeout"; kill $PID; exit 1; }
  sleep 5
done
echo "loaded in $(( $(date +%s) - t0 )) s"
pixi run -q python scripts/n6_bench.py http://127.0.0.1:$PORT $name results/night6/$name.json
rc=$?
kill $PID; sleep 5; pkill -P $PID 2>/dev/null; wait $PID 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep engines/ft | cut -d, -f1); do kill $p 2>/dev/null; done; sleep 5
exit $rc
