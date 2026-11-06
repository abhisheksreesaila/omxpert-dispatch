#!/bin/bash
# Mac chain: (1) batching prototype on Qwen3-Coder-30B, (2) gpt-oss-120b coding trace for its profile.
cd ~/expert-dispatch; B=llama.cpp/build/bin; O=results/mac; R=results
until [ -f $O/ppl_skip.done ]; do sleep 30; done
until [ "$(stat -f %z models/gptoss/gpt-oss-120b-MXFP4.gguf 2>/dev/null)" = "63387346208" ]; do sleep 30; done
echo "# $(date) download complete; batching prototype on Coder-30B" >> $O/batch.log
M=models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 48 -c 8192 -b 512 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8"
echo "## coder exact" >> $O/batch.log
perl -e 'alarm shift; exec @ARGV' 3600 $B/llama-batched-bench -m $M $C 2>&1 | grep -E "^\|" >> $O/batch.log
echo "## coder steer-only λ=1 (6 GB hot)" >> $O/batch.log
EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 perl -e 'alarm shift; exec @ARGV' 3600 $B/llama-batched-bench -m $M $C --moe-cache-mib 6144 2>&1 | grep -E "^\|" >> $O/batch.log
echo "# $(date) gpt-oss-120b trace (coding train sample)" >> $O/batch.log
perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-edx-trace -m models/gptoss/gpt-oss-120b-MXFP4.gguf -f $R/gptoss_train.txt -o $R/gptoss_trace.bin -nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36 -c 2048 -b 2048 -ub 2048 > $O/gptoss_trace.out 2>&1
echo "# $(date) trace exit $? $(tail -1 $O/gptoss_trace.out | cut -c1-150)" >> $O/batch.log; touch $O/chain3.done
