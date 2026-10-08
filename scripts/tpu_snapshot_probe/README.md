# TPU snapshot probe (Phase 0)

Scripts that reproduce the Phase 0 experiment from
[`TPU_SPOT_CHECKPOINTING_REPORT.md`](../../TPU_SPOT_CHECKPOINTING_REPORT.md):
snapshot a running vLLM server (CPU + TPU state) with gVisor and libtpu on one
Spot `tpu-v6e-8`, and restore it on another.

These are reconstructions of the scripts used in the October 6–7, 2026 runs. The
original working copies were not kept, so expect small fixes on the first rerun.

## Files

| File | Purpose |
| --- | --- |
| `tpu-probe.yaml` | SkyPilot setup for a probe VM: Docker, gVisor `runsc` (`runsc-tpu` runtime), all host workarounds, `tpucheckpoint`, libtpu 0.0.49 venv and vLLM image, model weights, RAM disks |
| `common.sh` | Shared settings (image, model, env vars, devices, `runsc` flags). Source and target must create identical containers |
| `probe.py` | Small JAX program that fills HBM and verifies it on device; honours `/probe/PAUSE` as a safe point |
| `trial_inplace.sh` | Step 3: in-place checkpoint/restore with `tpucheckpoint`, no gVisor |
| `probe_gvisor.sh` | Steps 4–5: `probe.py` inside gVisor; `start`, `checkpoint DIR [--leave-running]`, `restore DIR` |
| `start_vllm.sh` | Start vLLM inside gVisor and wait for health |
| `scen.py` | 100-request Responses-API client (`submit`, `report`) |
| `stage1_pause.sh` | Submit the burst, then `POST /pause?mode=keep&clear_cache=false` |
| `stage2_checkpoint.sh` | `runsc checkpoint` the paused server and copy the snapshot to the target VM |
| `stage3_restore.sh` | Restore on the target VM, wait for libtpu's restore ack, `POST /resume` |
| `stage4_verify.sh` | Poll to completion and compare with the snapshot-time state |

## Run order

```bash
# from this directory, on your machine
sky launch -c probe-a -i 600 --down tpu-probe.yaml
sky launch -c probe-b -i 600 --down tpu-probe.yaml
```

Then make the two VMs identical and connected:

1. Same gVisor build: compare `runsc --version`. If they differ, copy probe-a's
   `/usr/local/bin/runsc` and `/usr/local/bin/gvisor-bin/` to probe-b and restart Docker.
2. Same image digest: `sudo docker save vllm-tpu:gemma4-l49 | ssh <probe-b> sudo docker load`.
3. Transfer key: append probe-a's `~/.ssh/xfer.pub` to probe-b's `~/.ssh/authorized_keys`.
   On probe-a, use `ssh -F /dev/null -i ~/.ssh/xfer ...` (its `~/.ssh/config` may be invalid).

Probe-level tests (optional, before vLLM):

```bash
# on probe-a, in ~/sky_workdir
./trial_inplace.sh                                     # step 3
./probe_gvisor.sh start
./probe_gvisor.sh checkpoint /mnt/snap/ck1 --leave-running   # step 4a
./probe_gvisor.sh checkpoint /mnt/snap/ck2             # stops the sandbox
# copy /mnt/snap/ck2 to probe-b, then on probe-b:
./probe_gvisor.sh restore /mnt/snap/ck2                # step 5
```

vLLM scenario (step 6):

```bash
# probe-a
./start_vllm.sh                       # ~13-15 min
./stage1_pause.sh
./stage2_checkpoint.sh <probe-b internal IP>
# probe-b
./stage3_restore.sh                   # ~15 min through docker
./stage4_verify.sh
```

Long steps are best run detached on the VM (`nohup ./stageN.sh > stageN.log 2>&1 &`),
because SSH sessions to busy TPU VMs dropped several times during Phase 0.

Tear down when done: `sky down probe-a probe-b`.

## What each workaround is for

| Setting | Why |
| --- | --- |
| libtpu ≥ 0.0.48, `LIBTPU_CHECKPOINTING_ENABLED=true` | Older libtpu (the image ships 0.0.38) has no checkpoint protocol |
| `LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false` | The default HBM copy pipeline aborted with "DMA buffers were still outstanding" |
| `/pause?mode=keep&clear_cache=false` before checkpoint | A busy engine dies when libtpu detaches the TPUs; without `clear_cache=false` vLLM 0.17.2 crashes on TPU |
| `runsc checkpoint` directly | `docker checkpoint` pauses the container first and deadlocks libtpu's control thread |
| gVisor netstack + `VBAR_CONTROL_SERVICE_URL=<docker0>:8353` | Host-network sockets can't be saved (vLLM's gloo listener aborted); libtpu still needs the host TPU service |
| Wait for libtpu's restore ack before `/resume` | gVisor resumes app threads before libtpu has reattached the TPUs |
| Hide `/dev/vfio/devices`, raise `vm.max_map_count` | runsc device-scan bug; XLA compile under gVisor exceeds the default map limit |
| RAM disks for containerd content store and `TMPDIR` | `docker start --checkpoint` copies the whole image twice; the 97 GB boot disk can't hold it |
