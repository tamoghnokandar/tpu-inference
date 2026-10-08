#!/bin/bash
# Phase 0 steps 4-5: run probe.py inside gVisor and checkpoint/restore it.
#   probe_gvisor.sh start                start probe.py in the sandbox
#   probe_gvisor.sh checkpoint DIR [--leave-running]
#   probe_gvisor.sh restore DIR          on this or another VM (same runsc, image, /opt/probe)
# For a cross-VM restore, copy DIR to the target VM first.
set -u
source "$(dirname "$0")/common.sh"
C=probe-gv
ARGS=(--runtime=runsc-tpu $DEV --shm-size 16g -v /opt/probe:/probe
      -e VBAR_CONTROL_SERVICE_URL="${DOCKER0_IP}:8353"
      -e LIBTPU_CHECKPOINTING_ENABLED=true -e LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false
      --entrypoint /probe/v49/bin/python "$IMAGE" -u /probe/probe.py)

case "$1" in
  start)
    sudo rm -f /opt/probe/PAUSE; sudo docker rm -f $C >/dev/null 2>&1
    sudo docker run -d --name $C "${ARGS[@]}" >/dev/null
    for _ in $(seq 1 60); do sleep 5; sudo docker logs $C 2>&1 | grep -qE "step=2 |Traceback" && break; done
    sudo docker logs --tail 2 $C ;;
  checkpoint)
    sudo touch /opt/probe/PAUSE; sleep 7   # safe point
    T0=$(date +%s.%N)
    $RUNSC checkpoint --image-path="$2" ${3:-} "$(sudo docker inspect -f '{{.Id}}' $C)"
    echo "checkpoint took $(echo "$(date +%s.%N)-$T0" | bc)s, size $(sudo du -sh "$2" | cut -f1)"
    if [ "${3:-}" = --leave-running ]; then sudo rm -f /opt/probe/PAUSE; sleep 12; sudo docker logs --tail 2 $C; fi ;;
  restore)
    sudo touch /opt/probe/PAUSE; sudo rm -rf /tmp/runsc
    sudo docker rm -f $C >/dev/null 2>&1; sudo docker create --name $C "${ARGS[@]}" >/dev/null
    D=/var/lib/docker/containers/$(sudo docker inspect -f '{{.Id}}' $C)/checkpoints/ck
    sudo mkdir -p "$D" && sudo mount --bind "$2" "$D"
    sudo docker start --checkpoint=ck $C
    for _ in $(seq 1 600); do sudo sh -c "grep -h 'tpu_control.go:205' /tmp/runsc/*boot* 2>/dev/null" | grep -q success && break; sleep 1; done
    sudo rm -f /opt/probe/PAUSE; sleep 12
    sudo docker logs --tail 3 $C ;;   # expect the step counter to continue with hbm_match=True
esac
