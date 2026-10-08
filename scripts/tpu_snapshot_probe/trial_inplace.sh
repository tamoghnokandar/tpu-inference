#!/bin/bash
# Phase 0 step 3: in-place TPU checkpoint/restore with tpucheckpoint, no gVisor.
# Runs probe.py under plain Docker, pauses it at a safe point, detaches the TPUs
# (HBM -> host RAM), waits, restores, and checks HBM still matches.
set -u
NAME=${1:-inplace}
sudo rm -f /opt/probe/PAUSE
sudo docker rm -f probe-runc >/dev/null 2>&1
sudo docker run -d --name probe-runc --privileged --network host --shm-size 16g -v /opt/probe:/probe \
  -e LIBTPU_CHECKPOINTING_ENABLED=true -e LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false \
  --entrypoint /probe/v49/bin/python vllm/vllm-tpu:gemma4 -u /probe/probe.py >/dev/null
for _ in $(seq 1 40); do sleep 5; sudo docker logs probe-runc 2>&1 | grep -qE "step=2 |Error" && break; done

PID=$(sudo docker inspect -f '{{.State.Pid}}' probe-runc)
rss() { sudo awk '/VmRSS/{printf "%.2fGiB", $2/1048576}' "/proc/$PID/status"; }
echo "[$NAME] before: $(sudo docker logs --tail 1 probe-runc 2>&1) rss=$(rss)"
echo "[$NAME] state: $(sudo ~/tpucheckpoint --get-state --pid "$PID" 2>&1)"   # expect STATE_RUNNING

sudo touch /opt/probe/PAUSE; sleep 7                                            # safe point
T0=$(date +%s.%N); OUT=$(sudo ~/tpucheckpoint --action checkpoint --pid "$PID" 2>&1)
echo "[$NAME] checkpoint: $OUT ($(echo "$(date +%s.%N)-$T0" | bc)s)"
[ -d "/proc/$PID" ] || { echo DIED; sudo docker logs probe-runc 2>&1 | grep -E "^F[0-9]" | sort -u | cut -c1-200; exit 1; }
echo "[$NAME] detached: $(sudo ~/tpucheckpoint --get-state --pid "$PID" 2>&1) vfio_fds=$(sudo ls -la /proc/$PID/fd | grep -c vfio) rss=$(rss)"

sleep 15
T0=$(date +%s.%N); OUT=$(sudo ~/tpucheckpoint --action restore --pid "$PID" 2>&1)
echo "[$NAME] restore: $OUT ($(echo "$(date +%s.%N)-$T0" | bc)s)"
echo "[$NAME] state: $(sudo ~/tpucheckpoint --get-state --pid "$PID" 2>&1)"
sudo rm -f /opt/probe/PAUSE; sleep 16
echo "[$NAME] after resume:"; sudo docker logs --tail 4 probe-runc 2>&1 | sed 's/^/    /'   # expect hbm_match=True
