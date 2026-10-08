# TPU Spot snapshot: what changed and how it works

This branch adds a plan and test scripts for surviving Spot TPU preemption by
snapshotting a running vLLM server (CPU + TPU memory) and restoring it on a new VM.
No existing source code in `tpu_inference/` was changed.

## The method

vLLM runs inside a **gVisor** sandbox. gVisor acts as the kernel for vLLM, so it can save
the whole process to files. It cannot read TPU memory, so it asks **libtpu** to move TPU
memory (weights + KV cache) into normal RAM first. Restore runs the same steps in reverse.

```
Source VM                                                 Target VM
─────────                                                 ─────────
1. POST /pause?mode=keep&clear_cache=false
   (vLLM stops scheduling, keeps all requests + KV cache)
2. runsc checkpoint --image-path=DIR <container>
3.   gVisor → libtpu: ACTION_CHECKPOINT
4.   libtpu copies TPU HBM → host RAM, releases TPUs
5.   gVisor writes all memory/threads/sockets to DIR
6. copy DIR ───────────────────────────────────────────►  7. create identical container,
                                                             restore from DIR
                                                          8. gVisor rebuilds processes
                                                             (vLLM still paused)
                                                          9. gVisor → libtpu: ACTION_RESTORE
                                                             libtpu copies RAM → TPU HBM
                                                         10. wait for libtpu "success",
                                                             then POST /resume
```

Result of the test (Qwen3-30B-A3B, two Spot `tpu-v6e-8` VMs, 100 requests):

- 20 requests finished before the snapshot: output identical after restore (not rerun).
- 80 requests frozen mid-generation: all finished after restore.
- Prompt (prefill) throughput stayed at 0 after resume: requests continued from the
  restored KV cache, nothing was recomputed.
- Checkpoint 155 s, snapshot 92 GB, copy between VMs ~3 min, restore ~15 min through Docker.

Requests not in the snapshot (arrived during or after it) are lost; clients retry them.
Clients should use background mode (`POST /v1/responses` with `"background": true`, then
poll `GET /v1/responses/{id}`) so they can fetch results from the restored server.

### Required settings

| Setting | Why |
| --- | --- |
| libtpu ≥ 0.0.48 | Older versions have no checkpoint support. The `vllm/vllm-tpu:gemma4` image has 0.0.38; this repo pins 0.0.44. |
| `LIBTPU_CHECKPOINTING_ENABLED=true` | Turns on libtpu's checkpoint control channel. |
| `LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false` | The default copy path crashed. |
| `VLLM_SERVER_DEV_MODE=1` + `/pause?mode=keep&clear_cache=false` | Checkpointing a busy engine kills it. Without `clear_cache=false`, vLLM 0.17.2 crashes on TPU. |
| Call `runsc checkpoint` directly | `docker checkpoint` pauses the container first and deadlocks libtpu. |
| gVisor's own network stack (not `--network=host`) | Host sockets can't be saved; vLLM's internal PyTorch socket broke on restore. |
| `VBAR_CONTROL_SERVICE_URL=<docker0 IP>:8353` | Lets libtpu reach the host TPU service from gVisor's network stack. |
| Wait for libtpu's restore "success" before `/resume` | gVisor restarts app threads before libtpu has reattached the TPUs. |
| Move `/dev/vfio/devices` aside, `vm.max_map_count=4194304` | gVisor bug with that directory; XLA compile under gVisor needs more memory mappings. |
| Same runsc binary, image digest, and model paths on both VMs | gVisor requires an identical container to restore into. |

## Files

### `TPU_SPOT_CHECKPOINTING_REPORT.md`

The full plan and results.

- **Section 0:** Phase 0 results: setup, per-step results, timings, required settings, open items.
- **Section 1:** what a snapshot preserves, and which requests clients must retry.
- **Section 2:** acceptance gates, with current status.
- **Section 3:** design: process layout, how to take a snapshot, how to restore, snapshot size,
  proposed `tpu-test.yaml` changes.
- **Section 4:** phases. Phase 0 is done; Phase 1 is next.
- **Sections 5–6:** security, retention, risks.
- **Appendices:** external findings and repository findings.

### `scripts/tpu_snapshot_probe/`

Scripts to reproduce the test. They are rebuilt from the commands used in the test runs and
have not been run in this exact form, so expect small fixes on first use.

| File | What it does |
| --- | --- |
| `README.md` | Run order and what each workaround is for. |
| `tpu-probe.yaml` | SkyPilot setup for one probe VM: installs Docker and gVisor, registers the `runsc-tpu` runtime, applies the host fixes, builds `tpucheckpoint`, builds a vLLM image with libtpu 0.0.49, downloads the model, creates RAM disks for snapshots and Docker scratch space. |
| `common.sh` | Shared settings sourced by the other scripts: image, model, environment variables, TPU devices, `runsc` flags. Keeps both VMs' containers identical. |
| `probe.py` | Small JAX program that fills TPU memory with known data and checks it every 5 s. Stops TPU work while `/probe/PAUSE` exists. |
| `trial_inplace.sh` | Checkpoint and restore `probe.py` in place with `tpucheckpoint` (no gVisor). Checks libtpu support. |
| `probe_gvisor.sh` | Run `probe.py` inside gVisor: `start`, `checkpoint DIR [--leave-running]`, `restore DIR`. Used for same-VM and cross-VM tests. |
| `start_vllm.sh` | Start vLLM inside gVisor and wait until it is healthy (~13–15 min). |
| `scen.py` | Client that submits 100 background requests and reports their status. |
| `stage1_pause.sh` | Submit the requests, wait until ~20 finish, pause vLLM, record every request's state. |
| `stage2_checkpoint.sh` | Check vLLM is paused, run `runsc checkpoint`, copy the snapshot to the target VM. |
| `stage3_restore.sh` | Restore on the target VM, wait for libtpu's restore success, resume vLLM. |
| `stage4_verify.sh` | Wait for all requests to finish and compare with the snapshot-time state. |

## Not done yet

- Gemma 4 not tested (needs an HF token on the VM).
- Restore via `runsc restore` directly (expected to be faster than Docker, not measured).
- Periodic snapshots while serving (`--leave-running`) not measured with vLLM.
- No GCS upload/download, automatic recovery, or DNS switch yet.
- One trial only, same zone. The plan calls for 30 trials, including another zone.
