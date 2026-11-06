#!/bin/bash
# Expert Dispatch on unified memory (Apple M4, 16 GB): model (18.6 GB) > RAM, so experts page from SSD.
# All runs: no weight repacking (-nr) so CPU experts stay file-backed mmap pages (repack copies them to anon RAM -> swap).
# Each config: 1 warm-up prompt (not counted), then 4 coding prompts x 128 tokens. Logs ms/token, page-ins, RSS.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; R=results; O=results/mac; mkdir -p $O
to() { perl -e 'alarm shift; exec @ARGV' "$@"; }
run() { # name, env..., -- args...
  name=$1; shift; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for p in p09 p00 p01 p02 p03; do
    pi0=$(vm_stat | awk '/Pageins/ {gsub("\\.","",$2); print $2}')
    out=$( { /usr/bin/time -l env "${envs[@]}" X=1 perl -e 'alarm shift; exec @ARGV' 900 $B/llama-completion -m $M -f $R/prompts/$p.txt -n 128 -no-cnv --temp 0 -c 4096 --no-display-prompt -v -nr -fit off -lm mmap "$@" > /dev/null; } 2>&1 )
    pi1=$(vm_stat | awk '/Pageins/ {gsub("\\.","",$2); print $2}')
    ms=$(echo "$out" | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
    pp=$(echo "$out" | grep "prompt eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
    rss=$(echo "$out" | awk '/maximum resident set size/ {print int($1/1048576)}')
    edx=$(echo "$out" | grep -E "EDX (pinned|.*took|majflt)" | tr '\n' ' ' | cut -c1-200)
    echo "$name $p ms=$ms pp_ms=$pp pageins_16k=$((pi1-pi0)) rss_mib=$rss $edx" >> $O/bench.log
    [ -z "$ms" ] && echo "$out" | tail -15 > $O/fail_${name}_$p.log
  done
}
echo "# $(date) start" >> $O/bench.log
run full_metal -- -ngl 99
run experts_cpu -- -ngl 99 --n-cpu-moe 48
run layers24 -- -ngl 99 --n-cpu-moe 24
run slru4g -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 4096
run pin4g EDX_PIN=$R/pin_coding.bin -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 4096
run pin4g_pageout EDX_PIN=$R/pin_coding.bin EDX_PAGEOUT=1 -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 4096
run pin6g_pageout EDX_PIN=$R/pin_coding.bin EDX_PAGEOUT=1 -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 6144
run pin6g_pageout_steer1 EDX_PIN=$R/pin_coding.bin EDX_PAGEOUT=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 6144
run pin6g_pageout_chatprofile EDX_PIN=$R/pin_chat.bin EDX_PAGEOUT=1 -- -ngl 99 --n-cpu-moe 48 --moe-cache-mib 6144
echo "# $(date) done" >> $O/bench.log; touch $O/bench.done
