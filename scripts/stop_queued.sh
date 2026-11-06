#!/bin/bash
for p in $(pgrep -f "bias_chat_after[.]sh|scripts/run_max[.]sh"); do kill $p; done
