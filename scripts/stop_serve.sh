#!/bin/bash
for p in $(pgrep -f "bin/max serve"); do kill $p; done
for p in $(pgrep -f "Model Worker\|max._entrypoints\|max serve"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
for i in $(seq 30); do pgrep -f "bin/max serve" >/dev/null || break; sleep 1; done
# model workers are multiprocessing-spawn children; kill anything from the MAX env
for p in $(pgrep -f "expert-dispatch/max/[.]pixi/envs/default/bin/python"); do kill $p 2>/dev/null; done
for i in $(seq 30); do pgrep -f "expert-dispatch/max/[.]pixi/envs/default/bin/python" >/dev/null || break; sleep 1; done
for p in $(pgrep -f "expert-dispatch/max/[.]pixi/envs/default/bin/python"); do kill -9 $p 2>/dev/null; done
