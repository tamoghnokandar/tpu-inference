#!/bin/bash
# Start vLLM inside the gVisor sandbox (source VM) and wait until it is healthy.
# Cold start under gVisor took 13-15 min in Phase 0 (mostly XLA compilation).
set -u
source "$(dirname "$0")/common.sh"

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1
sudo rm -rf /tmp/runsc
sudo docker run -d --name "$CONTAINER" "${VLLM_DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_SERVE_ARGS[@]}" >/dev/null

T0=$(date +%s)
until curl -sf -m 10 localhost:8000/health >/dev/null; do
  [ "$(sudo docker inspect -f '{{.State.Running}}' "$CONTAINER")" = true ] || { echo "vLLM exited"; exit 1; }
  sleep 10
done
echo "vLLM ready after $(( $(date +%s) - T0 ))s"
