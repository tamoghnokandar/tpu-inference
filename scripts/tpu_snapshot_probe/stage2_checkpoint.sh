#!/bin/bash
# Stage 2 (source VM): stop-checkpoint the paused vLLM sandbox and copy the
# snapshot to the target VM.   usage: stage2_checkpoint.sh TARGET_INTERNAL_IP
set -u
source "$(dirname "$0")/common.sh"
PEER=$1

[ "$(curl -s -m 10 localhost:8000/is_paused)" = '{"is_paused":true}' ] || { echo "vLLM is not paused; refusing to checkpoint"; exit 1; }

# gVisor sends ACTION_CHECKPOINT to libtpu, which copies HBM into host memory and
# releases the TPUs; gVisor then writes checkpoint.img, pages.img, pages_meta.img.
T0=$(date +%s.%N)
$RUNSC checkpoint --image-path=/mnt/snap/v1 "$(container_id)" 2>&1 | tail -2
echo "checkpoint took $(echo "$(date +%s.%N)-$T0" | bc)s; size $(sudo du -sh /mnt/snap/v1 | cut -f1)"
sudo sh -c "grep -hE 'tpu_control.go:(140|205)|Checkpoint attempt failed' /tmp/runsc/*boot* 2>/dev/null | sort | tail -2"

[ -s /mnt/snap/v1/pages.img ] || { echo "no snapshot written"; exit 1; }
sudo chmod -R a+rX /mnt/snap/v1
O="-F /dev/null -i $HOME/.ssh/xfer -o StrictHostKeyChecking=accept-new"
ssh $O "gcpuser@$PEER" "sudo rm -rf /mnt/snap/v1; sudo mkdir -p /mnt/snap/v1; sudo chmod 777 /mnt/snap /mnt/snap/v1"
T0=$(date +%s)
scp -q $O /mnt/snap/v1/* "gcpuser@$PEER:/mnt/snap/v1/" && scp -q $O /mnt/snap/*.json "gcpuser@$PEER:/mnt/snap/"
echo "copied to $PEER in $(( $(date +%s) - T0 ))s"
