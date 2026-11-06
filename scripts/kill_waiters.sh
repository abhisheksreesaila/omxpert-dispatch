#!/bin/bash
for p in $(pgrep -f "scripts/run_(max|bias)[.]sh"); do kill $p; done
