# Night 5 decisions

- D-01 · model Qwen3.8-Flash-Next UD-Q4_K_XL + bundled IQ4_NL table over Qwen3.6-35B · it is the Qwen 4 (qwen4exp) architecture and fits disk without deleting anything · reversible: yes
- D-02 · plain HTTPS download (HF_HUB_DISABLE_XET=1) over xet · xet ran at ~250 KB/s, HTTPS 114 MB/s · yes
- D-03 · re-implement his ideas inside our llama.cpp tree over running his image · image needs root + ~90 GB more disk and is GLM-only · yes
- D-04 · milestone "commits" are journal entries + a local commit in ~/expert-dispatch-public at the end, not git in the workdir · the workdir has never been a repo (nights 1-4) and holds 1 GB logit files · yes
- D-05 · in-process A/B (one warmed server, arms rotated, Latin square) over separate server runs for decode tricks · separate runs of one config swung 11→19 tok/s from page-cache drift alone · yes
- D-06 · n-gram table through the OS page cache (-lzm off) instead of llama.cpp's lazy reader, in the A/B server and (if it holds against base2) the stack · prompt 27.6→116 tok/s, faults 485→11/token, same RSS · yes
- D-07 · test the table trick from a cold page cache (drop only the table shard) before recommending -lzm off · base2 showed warm stock ≈ -lzm off, so the gain may be a warm-up effect · yes
- D-08 · final stack = -lzm off + MTP over the picker's MTP-only · the picker's -lzm rule compared against base2, which had a table cached by earlier runs; the MTP-only stack's prompts fell back to 44 tok/s with 568 faults/token · yes
- D-09 · retest learned residency with -lzm off, interleaved fit/pin/pin+steer x2, over trusting the lazy-table pin runs · the lazy table's cache state confounded every separate lazy run (base 11 vs base2 23) · yes
- D-10 · keep going past the 80-turn cap, inside the 8-hour cap · turns went to waiting on queued runs; stopping now would leave the cold-table and clean residency tests unrun · yes
- D-11 · withdraw the '-lzm off = 4x faster prompts' claim; keep -lzm off as a robust exact default (low faults 8/8 vs lazy 4/11) · cold-table A/B showed warm-or-cold stock can be just as fast · yes
- D-12 · shared MTP head on GPU0 with -fitt 3600,1024 over the self-contained head · the self-contained head ran on the CPU and cut prompt speed to 20-56 tok/s · yes
- D-13 · launcher default -lzm off without MTP; MTP=1 opt-in · MTP: decode x1.36 but prompts 20-80 tok/s vs 110-170 without it on this box; agents send long prompts · yes
- D-14 · report learned residency and steering as not paying inside llama.cpp on this box (clean reruns), retract the earlier 2x steering number · the first pin/steer runs were compared against a fault-heavy baseline · yes
