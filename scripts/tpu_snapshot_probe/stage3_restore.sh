#!/bin/bash
# Stage 3 (target VM): restore /mnt/snap/v1 into an identical container, wait for
# libtpu to reload HBM, then resume vLLM.
set -u
HERE="$(dirname "$0")"
source "$HERE/common.sh"

sudo rm -rf /tmp/runsc
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1
sudo docker create --name "$CONTAINER" "${VLLM_DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_SERVE_ARGS[@]}" >/dev/null

# docker start --checkpoint copies the image into containerd's content store and
# then into $TMPDIR before calling runsc restore; both need ~image-size free space
# (tpu-probe.yaml puts them on RAM disks). Calling runsc restore directly should be
# faster but was not measured in Phase 0.
D=/var/lib/docker/containers/$(container_id)/checkpoints/v1
sudo mkdir -p "$D" && sudo mount --bind /mnt/snap/v1 "$D"
T0=$(date +%s.%N)
sudo docker start --checkpoint=v1 "$CONTAINER" 2>&1 | tail -3
echo "docker start --checkpoint took $(echo "$(date +%s.%N)-$T0" | bc)s"

# gVisor resumes app threads before libtpu has reattached the TPUs, so vLLM must
# stay paused until libtpu acknowledges ACTION_RESTORE.
for _ in $(seq 1 900); do
  sudo sh -c "grep -h 'tpu_control.go:205' /tmp/runsc/*boot* 2>/dev/null" | grep -q success && break
  sleep 1
done
sudo sh -c "grep -hE 'tpu_control.go:(140|205)|FATAL ERROR' /tmp/runsc/*boot* 2>/dev/null | sort | tail -3"
echo "libtpu restore ack $(echo "$(date +%s.%N)-$T0" | bc)s after start"

for _ in $(seq 1 60); do curl -sf -m 5 localhost:8000/health >/dev/null && break; sleep 1; done
echo "health: $(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:8000/health)  is_paused: $(curl -s -m 10 localhost:8000/is_paused)"
python3 "$HERE/scen.py" report /mnt/snap/after_restore.json
echo "resume: $(curl -s -m 60 -X POST localhost:8000/resume)"
