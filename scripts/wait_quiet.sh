#!/bin/bash
# block until 1-min load < 3 for 3 checks in a row (60 s apart), so timings are clean and the box isn't overloaded
n=0; while [ $n -lt 3 ]; do l=$(cut -d' ' -f1 /proc/loadavg); if awk "BEGIN{exit !($l < 3)}"; then n=$((n+1)); else n=0; fi; sleep 60; done
