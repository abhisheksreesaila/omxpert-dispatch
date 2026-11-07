#!/bin/bash
# Night 5 queue 5 (after queue 4): cold-table test. Drop only the n-gram table shard from the page cache before each run, then
# stock (lazy table) vs -lzm off (table mapped normally and populated at load). base (cold, first run of the day) gave 11.0 tok/s,
# base2 (after many -lzm off runs had cached the table) 23.0: is the table's cache state the difference?
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue4.done ]; do sleep 30; done
TABLE=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00004-of-00004.gguf)
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
drop() { python3 -c "import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" "$TABLE"; }
run() { [ -f $R/$1.json ] && return; drop; log "run $1 cold table ($ARGS)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
ARGS="" EDXENV="" run cold_lazy_1
ARGS="-lzm off" EDXENV="" run cold_lzm_1
ARGS="" EDXENV="" run cold_lazy_2
ARGS="-lzm off" EDXENV="" run cold_lzm_2
log "queue5 done"; touch $R/queue5.done
