#!/bin/bash
# Shared settings for the Phase 0 snapshot probe. Source this file; do not run it.
# The source and target VMs must create the vLLM container with exactly these
# arguments, because gVisor validates the restored container spec.

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
IMAGE="${IMAGE:-vllm-tpu:gemma4-l49}"   # vllm/vllm-tpu:gemma4 + libtpu 0.0.49 (built in setup)
CONTAINER=vllm-gv

# TPU devices for gVisor's tpuproxy: the numbered VFIO groups plus the container
# device. /dev/vfio/devices must already be moved aside (see tpu-probe.yaml).
DEV="--device /dev/vfio/vfio $(for d in /dev/vfio/[0-9]*; do printf -- '--device %s ' "$d"; done)"

# Host TPU service, reached from gVisor's netstack through the docker bridge.
DOCKER0_IP=$(ip -4 addr show docker0 | awk '/inet /{print $2}' | cut -d/ -f1)

VLLM_DOCKER_ARGS=(
  --runtime=runsc-tpu $DEV -p 8000:8000 --shm-size 16g
  -v /mnt/hf30:/hf -v /mnt/xla:/xla
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_XLA_CACHE_PATH=/xla
  -e VBAR_CONTROL_SERVICE_URL="${DOCKER0_IP}:8353"
  -e LIBTPU_CHECKPOINTING_ENABLED=true
  -e LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false
  -e MODEL_IMPL_TYPE=vllm                 # Qwen3 on this image's transformers needs vLLM's model code
  -e VLLM_ENABLE_RESPONSES_API_STORE=1    # background /v1/responses, retrievable after restore
  -e VLLM_SERVER_DEV_MODE=1               # exposes /pause, /resume, /is_paused
  --entrypoint vllm
)
VLLM_SERVE_ARGS=(
  serve "$MODEL" --host 0.0.0.0 --port 8000
  --max-model-len 4096 --tensor-parallel-size 8 --gpu-memory-utilization 0.4
)

# runsc must be called directly (docker checkpoint pauses the container first and
# deadlocks libtpu's control thread). Top-level flags must match the runtime's.
RUNSC="sudo runsc --root /var/run/docker/runtime-runc/moby --tpuproxy --debug --debug-log=/tmp/runsc/"

container_id() { sudo docker inspect -f '{{.Id}}' "$CONTAINER"; }
