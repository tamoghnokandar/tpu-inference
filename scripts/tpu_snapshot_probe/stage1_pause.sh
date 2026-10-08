#!/bin/bash
# Stage 1 (source VM): submit 100 background requests; once ~20 have finished,
# freeze vLLM at a safe point and record every request's state.
set -u
HERE="$(dirname "$0")"

sudo rm -rf /mnt/snap/v1 /mnt/snap/*.json
sudo chmod 777 /mnt/snap
python3 "$HERE/scen.py" submit 20

# mode=keep freezes running and queued requests with their KV cache.
# clear_cache=false is required: vLLM 0.17.2 clears caches even in keep mode
# otherwise, which crashes the TPU worker (reset_encoder_cache not implemented).
echo "pause: $(curl -s -m 120 -X POST 'localhost:8000/pause?mode=keep&clear_cache=false')"
echo "is_paused: $(curl -s -m 10 localhost:8000/is_paused)"

python3 "$HERE/scen.py" report /mnt/snap/at_pause.json
sleep 5
python3 "$HERE/scen.py" report /mnt/snap/at_snapshot.json
python3 - <<'P'
import json
a = json.load(open("/mnt/snap/at_pause.json")); b = json.load(open("/mnt/snap/at_snapshot.json"))
print("requests whose state/length changed during 5s pause:", sum(1 for k in a if a[k][:2] != b[k][:2]))
P
echo "engine: $(sudo docker logs vllm-gv 2>&1 | grep -oE 'generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs, Waiting: [0-9]+ reqs' | tail -2 | tr '\n' '|')"
