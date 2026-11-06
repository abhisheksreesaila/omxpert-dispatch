#!/bin/bash
for p in $(pgrep -f "scripts/run_max[.]sh"); do kill $p; done
