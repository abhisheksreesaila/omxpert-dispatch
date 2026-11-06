#!/bin/bash
# Night 3 on the Mac: how many experts a learned profile needs, whether the right profile matters, and how little history it takes.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/gptoss/gpt-oss-120b-MXFP4.gguf; R=results; O=results/mac; L=$O/night3.log
PY=/Library/Developer/CommandLineTools/usr/bin/python3
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36"
until [ -f $O/work.done ]; do sleep 60; done
SLOT=12.62   # MiB per expert slot in the cache budget (6144 MiB = 486 slots)
budget() { $PY -c "print(int(($1*36+1)*$SLOT)+4)"; }
speed() { name=$1; n=$2; shift 2; tot=0; for i in 0 1 2; do
    out=$(env "$@" X=1 perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-completion -m $M -f $R/short_$i.txt -n 96 -no-cnv --temp 0 --no-display-prompt -v $C -c 4096 --moe-cache-mib $(budget $n) 2>&1 >/dev/null)
    ms=$(echo "$out" | perl -ne 'print $1 if /^\S+ I common_perf_print:\s+eval time.*\(\s*([0-9.]+) ms per token/')
    pin=$(echo "$out" | perl -ne 'print "$1" if /EDX pinned (\d+) experts/')
    echo "speed $name short_$i ms=$ms pinned=$pin" >> $L; done; }
speed0() { for i in 0 1 2; do out=$(perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-completion -m $M -f $R/short_$i.txt -n 96 -no-cnv --temp 0 --no-display-prompt -v $C -c 4096 2>&1 >/dev/null)
    echo "speed exact short_$i ms=$(echo "$out" | perl -ne 'print $1 if /^\S+ I common_perf_print:\s+eval time.*\(\s*([0-9.]+) ms per token/')" >> $L; done; }
ppl() { name=$1; n=$2; shift 2
  out=$(env "$@" X=1 perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-perplexity -m $M -f $R/coding_heldout.txt -c 512 --chunks 4 -b 8 -ub 8 $C --moe-cache-mib $(budget $n) 2>&1)
  echo "ppl $name $(echo "$out" | perl -ne 'print $1 if /Final estimate: PPL = ([0-9.]+)/')" >> $L; }
H() { echo "EDX_PIN=$1 EDX_STEER_ONLY=1 EDX_PIN_MINPL=$2 EDX_STEER=$3"; }
P=$R/pin_gptoss_coding.bin
echo "# $(date) night3 start" >> $L
speed0
# 1. how many experts per layer does the learned profile need?
for n in 4 8 16 32; do for lam in 1 0.5; do speed n${n}_l$lam $n $(H $P $n $lam); ppl n${n}_l$lam $n $(H $P $n $lam); done; done
# 2. the right profile vs a wrong one: learn a chat profile (20 min), steer coding work toward it
echo "# $(date) chat trace" >> $L
perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-edx-trace -m $M -f $R/chat_train.txt -o $R/gptoss_chat_trace.bin $C -c 2048 -b 2048 -ub 2048 > $O/chat_trace.out 2>&1
$PY scripts/mac_mkpin.py $R/gptoss_chat_trace.bin $R/pin_gptoss_chat.bin >> $L 2>&1
$PY scripts/mac_mkpin.py $R/gptoss_trace.bin $R/pin_gptoss_cod1k.bin 1000 >> $L 2>&1
$PY scripts/mac_mkpin.py $R/gptoss_trace.bin $R/pin_gptoss_cod5k.bin 5000 >> $L 2>&1
$PY scripts/mac_mkpin.py $R/gptoss_trace.bin $R/pin_gptoss_cod25k.bin >> $L 2>&1
for lam in 1 0.5; do speed chatpin_n16_l$lam 16 $(H $R/pin_gptoss_chat.bin 16 $lam); ppl chatpin_n16_l$lam 16 $(H $R/pin_gptoss_chat.bin 16 $lam); done
# 3. learning curve: profiles from 1k and 5k tokens of history (the full one is 25k)
for p in cod1k cod5k; do speed ${p}_n16_l1 16 $(H $R/pin_gptoss_$p.bin 16 1); ppl ${p}_n16_l1 16 $(H $R/pin_gptoss_$p.bin 16 1); done
echo "# $(date) night3 done" >> $L; touch $O/night3.done
