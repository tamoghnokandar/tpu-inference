# TPU inference snapshot and restore on Google Cloud Spot VMs

**Prepared:** October 1, 2026; **plan revised:** October 5, 2026 (gVisor/libtpu snapshot is the primary design; no external gateway); **Phase 0 executed:** October 6–7, 2026 (results in Section 0)  
**Repository reviewed:** `vllm-project/tpu-inference`, commit `68e5a055b31808c246e95fc2967ffc12d9906ecb`  
**Deployment under consideration:** `tpu-test.yaml`: `vllm serve google/gemma-4-26B-A4B-it` on Spot `tpu-v6e-8` (`gcp/asia-south1`), TP=8, `max-model-len 16384`, image `vllm/vllm-tpu:gemma4`, launched with SkyPilot.  
**Objective:** Periodically snapshot the complete vLLM CPU + TPU state with gVisor (`runsc`) and libtpu's checkpoint protocol. After a Spot preemption, restore the latest complete snapshot on a replacement VM. The engine then continues from the snapshot point: finished requests stay finished, partially decoded requests continue from their snapshotted position, and queued requests are still queued. No gateway or external request journal is used. Requests the snapshot cannot carry over are rejected or dropped, and **clients retry them**.

**Status:** **Phase 0 (runtime probe) passed on real hardware.** A live vLLM server was checkpointed mid-burst on one Spot `tpu-v6e-8` and restored on a different one. Requests that had finished stayed finished, and requests frozen mid-decode completed after restore without recomputation (Section 0). The result comes from **one** end-to-end trial, with `Qwen/Qwen3-30B-A3B` rather than Gemma 4, greedy decoding, and same-zone VMs. It is a feasibility result, not a qualified production procedure. No snapshot agent, GCS upload, DNS switch, or managed recovery has been built yet. Timings in Section 0 are measured; all other budgets are still proposed criteria.

## 0. Phase 0 results (measured October 6–7, 2026)

### Test setup

| Item | Value |
| --- | --- |
| Hardware | Two Spot `tpu-v6e-8` VMs (`asia-south1-c`), each with 8 × TPU v6e ("v6 lite") and 1.4 TB host RAM; 97 GB boot disk |
| gVisor | `runsc` master build `release-20260928.0-265` (identical binary on both VMs), Docker runtime with `--tpuproxy` |
| Container image | `vllm/vllm-tpu:gemma4` (vLLM 0.17.2, JAX 0.9.2) with libtpu replaced by **0.0.49** (`pip install --no-deps libtpu==0.0.49`); identical image digest copied to both VMs |
| Model and server | `Qwen/Qwen3-30B-A3B`, TP=8, `--max-model-len 4096`, `--gpu-memory-utilization 0.4`, `MODEL_IMPL_TYPE=vllm`, Responses API store enabled |
| Workload | 100 Responses-API background requests, greedy (`temperature: 0`), `max_output_tokens` 300–1650 |

### Results by step

| Step | Result |
| --- | --- |
| 1. libtpu version in the image | `vllm/vllm-tpu:gemma4` ships **libtpu 0.0.38**, which has **no** checkpoint protocol. The repo's pinned 0.0.44 and every release through 0.0.47 lack it too. **libtpu 0.0.48 and 0.0.49 contain gVisor's `cloud_gvisor.tpu` control protocol** (LOCK / CHECKPOINT / RESTORE / UNLOCK / GETSTATE). It is off by default and switched on with `LIBTPU_CHECKPOINTING_ENABLED=true`. |
| 2–3. In-place checkpoint/restore, no gVisor (`tpucheckpoint`) | ✅ Passed. The control thread appeared (Linux truncated its name to `libtpu044a044b`), and `--get-state` returned `STATE_RUNNING`. Checkpoint took 6.4 s and left the process `DETACHED` with 0 VFIO fds and the HBM contents in host RAM. Restore took 7.7 s, and an on-device HBM checksum matched afterwards. Two conditions were required: `LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false` (the default pipeline aborted with "1 DMA buffers were still outstanding"), and no TPU work issued while detached. |
| 4. Inside gVisor, same VM | ✅ Passed with `runsc checkpoint` called directly. `--leave-running`: 42.6 s total pause, 7.0 GB image. Stop and restore: 18 s checkpoint, 62 s `docker start --checkpoint`. The program continued its step counter with HBM intact. |
| 5. Restore on a **different** VM | ✅ Passed. A test program paused at step 11 on probe-a resumed at step 12 on probe-b, with HBM bit-identical. The 5.0 GB image copied VM-to-VM in 9 s. libtpu restore acknowledgement took 39.5 s, and the whole restore 50.3 s. |
| 6. vLLM, 100-request scenario across VMs | ✅ Passed (one trial). At the snapshot, 20 requests were finished and 80 were frozen mid-decode (throughput 3,076 tok/s before the pause, 0 tok/s while paused). After restore on probe-b and `/resume`: all **20 finished texts were byte-identical** (not rerun), and all **80 frozen requests completed**. Prompt (prefill) throughput stayed at **0.0 tok/s** throughout, so decoding continued from the restored KV cache with no recomputation. |

### Measured timings (vLLM, step 6)

| Phase | Time / size |
| --- | --- |
| vLLM cold start under gVisor (weights on RAM disk) | 13–15 min, mostly XLA compilation |
| `POST /pause?mode=keep&clear_cache=false` | under 1 s; no request state changed during a 5 s check |
| `runsc checkpoint` (stop) | **155 s** total; libtpu `ACTION_CHECKPOINT` acknowledged after 111–115 s |
| Snapshot size | **92–95 GB** at `--gpu-memory-utilization 0.4` (≈ 102 GB of HBM in use) |
| VM-to-VM copy (internal network, `scp`) | 183–191 s (≈ 500 MB/s) |
| `docker start --checkpoint` (restore) | 756 s; Docker copies the image into containerd's content store and then to a temporary directory before `runsc restore` runs |
| libtpu `ACTION_RESTORE` | 165 s after the sandbox restored |
| Remaining 80 requests after `/resume` | finished in about 25 s |

The restore time is dominated by Docker's extra copies, not by gVisor or libtpu. Calling `runsc restore` directly should remove most of the 756 s; this is Phase 1 work.

### Required configuration (all found during Phase 0)

1. **libtpu ≥ 0.0.48** with `LIBTPU_CHECKPOINTING_ENABLED=true` and `LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false`. Swapping libtpu 0.0.49 into the vLLM 0.17.2 / JAX 0.9.2 image worked for both Qwen3 models tested.
2. **Engine safe point through vLLM's built-in endpoint:** `POST /pause?mode=keep&clear_cache=false` before the checkpoint and `POST /resume` after the restore. The endpoint only exists with `VLLM_SERVER_DEV_MODE=1`. Checkpointing a busy engine kills it (`JaxRuntimeError: CANCELLED: Cancelled by TearDown`). Despite its docstring, `mode=keep` still clears caches unless `clear_cache=false` is passed, and clearing crashes the TPU worker (`reset_encoder_cache` is not implemented).
3. **Call `runsc checkpoint` directly, not `docker checkpoint`.** Docker sends `containerManager.Pause` first, which freezes libtpu's control thread; gVisor's `ACTION_CHECKPOINT` then times out after 3 minutes and the process crashes.
4. **gVisor's own network stack (netstack), not `--network=host`.** With host networking, vLLM's PyTorch gloo listener socket is reset on restore and the process aborts (`gloo::EnforceNotMet … accept: Software caused connection abort`). Under netstack, run the container on Docker's bridge network with the port published, and set **`VBAR_CONTROL_SERVICE_URL=169.254.123.1:8353`** (the host's `docker0` address) so libtpu can reach the host TPU service. The metadata server is reachable from netstack.
5. **Keep the workload paused until libtpu acknowledges `ACTION_RESTORE`.** gVisor resumes application threads while libtpu is still reattaching the TPUs; TPU work in that window crashes the process (`TpuVxcDriver::WriteToSharedMemory`).
6. **Host settings for gVisor on TPU VMs:**
   - **Hide `/dev/vfio/devices`.** runsc treats every entry in `/dev/vfio` as an IOMMU group and fails on that directory. This is a gVisor bug, worked around by moving the directory aside.
   - **Raise `vm.max_map_count`** (to 4,194,304). XLA compilation under gVisor exceeded the default 65,530 mappings and segfaulted.
   - **Put containerd's content store and `TMPDIR` on a large disk or RAM disk.** Docker's restore path makes two full copies of the image.
7. **Identical software on source and target:** the same runsc binary, the same image digest, and model files at the same host paths. Host-mounted files (`-v`) are not inside the snapshot.

### Not yet verified

- Gemma 4 itself; Gemma 4 depends on the image's JAX model code path, while the Qwen3 runs needed `MODEL_IMPL_TYPE=vllm`.
- Output equality against an uninterrupted reference run; only "finished texts unchanged" and "no prefill after resume" were checked.
- Sampled decoding and RNG-state continuity.
- `--leave-running` with vLLM (periodic snapshots while serving); measured only with the test program.
- Restores in a different zone; the 30-trial gate; GCS upload/download; direct `runsc restore`; DNS switching; managed recovery after a real preemption.

## 1. Decision and recovery contract

### What a snapshot preserves

vLLM keeps all request state in memory: the API server's open requests, the scheduler's waiting and running queues, KV-cache blocks in HBM, the sampling RNG state, and the token outputs produced so far. A `runsc checkpoint` of the sandbox, with libtpu moving HBM contents into host memory first, captures all of it together. Restoring that image recreates the same engine at the same point.

Take your scenario: 100 requests are sent. At the snapshot boundary, 20 are finished, request 21 is mid-decode, and 79 are waiting. After restore:

- requests 1–20 are finished and are not rerun;
- request 21 continues decoding from the exact position it reached at the snapshot boundary, with its KV cache and RNG state intact;
- requests 22–100 are still in vLLM's waiting queue and are processed in the scheduler's normal order.

Unlike request replay, this preserves the **exact engine state**, including RNG. A sampled continuation can therefore match an uninterrupted run, if the restored execution order matches (Section 2).

### What a snapshot cannot preserve: the retry contract

A snapshot is a point in the past, and a restored VM has a new network identity. Without an external gateway, the following requests are **not** carried over. Clients must retry them:

| Category | What happens | Client sees |
| --- | --- | --- |
| **A. Arrives while a snapshot is being taken** | Rejected at the VM's firewall during the capture window (Section 3.2). It never enters vLLM, so it is not in any snapshot. | Immediate connection refusal, or `503` if the server is reachable. **Retry.** |
| **B. Arrives after the last complete snapshot, before preemption** | It reached the old VM but is missing from the snapshot that gets restored. | Connection drops at preemption. **Retry.** |
| **C. Was in the snapshot, client streaming over a live HTTP connection** | The engine state is restored, but the client's TCP connection belonged to the dead VM and is gone. | Connection drops at preemption. Must use the result-retrieval path below, or **retry from scratch** (the restored work is then wasted). |
| **D. Was in the snapshot, client submitted asynchronously (background mode)** | The restored server still holds the request and its result. | Client polls the new endpoint with its request/response ID and gets the result. **No retry needed.** |

So only category D benefits fully from snapshot restore. To get the behaviour you described (request 21 resumes, 22–100 complete without resubmission), clients must submit in a **resumable, asynchronous way** rather than over a single long-lived stream:

- **Recommended:** use vLLM's OpenAI Responses API in background mode (`POST /v1/responses` with `"background": true`, then `GET /v1/responses/{id}`), with the response store enabled (`VLLM_ENABLE_RESPONSES_API_STORE=1`). The response store lives in the API-server process, so it is captured and restored with the snapshot. After restore, clients poll the same response ID at the new endpoint and receive the finished output. **Confirmed in Phase 0** with vLLM 0.17.2: all 100 background response IDs were retrievable on the restored VM. Note: Responses API reports in-flight requests as `queued` until they finish, so it cannot show partial progress. The store is in memory only and grows until entries are removed.
- **Streaming clients (`/v1/chat/completions` with `stream: true`)** always fall into category C when the VM is lost. They must retry from scratch. If they had already received tokens generated *after* the snapshot boundary, a retry can produce different text when sampling is used. Clients should discard partial output on a disconnect.

Client retries must use exponential backoff with jitter, and must honour `Retry-After` when present. Document this as part of the service's API contract.

### Why snapshots are periodic, not taken at preemption time

- A v6e-8 has 8 × 32 GiB = 256 GiB of HBM. Gemma-4-26B-A4B weights in bf16 take about 52 GB, and vLLM pre-allocates most of the remaining HBM as KV cache [Worker memory sizing][repo-memory]. A full image can therefore be up to roughly 250 GiB plus host process memory. **Measured:** Qwen3-30B-A3B at `--gpu-memory-utilization 0.4` produced a 92–95 GB image, and the checkpoint took about 155 s.
- Spot preemption gives a short notice (Compute Engine documents about 30 seconds [Compute Engine preemption][compute-spot]; verify for the TPU API). Copying HBM to host and writing that much data to durable storage takes minutes, not seconds.
- So the design takes snapshots every `I` minutes, and recovery rolls back to the **last complete** one. Up to one interval of work is redone, and category B requests are lost. The preemption notice is used only to stop admitting new requests and, if capacity allows, to start a best-effort final snapshot, never as a correctness requirement.

## 2. Proposed acceptance gates

Freeze these thresholds before hardware benchmarking. They may be revised **before** measurement. They are not production promises.

| Gate | Proposed prototype pass criterion |
| --- | --- |
| Runtime support | `tpucheckpoint --get-state` reports a working control channel for the vLLM worker process. runsc checkpoint/restore acknowledgements are observed. Selected HBM buffers are bit-identical after restore. **Missing support stops the project.** **Met in Phase 0** (libtpu ≥ 0.0.48). |
| Cross-VM restore | Restore succeeds on a **different** v6e-8 VM (same zone first, then a different zone in `asia-south1`). **Same zone met in Phase 0; different zone not yet tested.** |
| 100-request scenario | Submit 100 background requests, snapshot mid-burst, kill the VM, restore elsewhere. Every request present in the snapshot completes and is retrievable by ID. Requests 1–N finished at the snapshot are not recomputed. Partially decoded requests continue from their snapshotted position. Pass in ≥ 30 trials with varied snapshot points. **1 of 30 trials run, and it passed.** |
| Output equality | Greedy: the restored run's outputs equal an uninterrupted reference run. Sampled: equal when no new requests arrive after restore (same execution order). Divergence is reported with its first divergent position. |
| Retry contract | Categories A–C clients receive a fast failure (refused, `503`, or reset), never a hang longer than the client timeout. No partial response is presented as complete. |
| Capture pause | p95 serving pause per snapshot ≤ a target set in Phase 0 from measured HBM→host copy rates. Report the full pause including source resume under `--leave-running`. |
| Snapshot freshness | Latest complete snapshot age ≤ 2 × `I` during healthy operation. |
| Recovery time | Report separately: capacity wait; snapshot download; `runsc restore` + HBM rehydration; endpoint switch; first useful output. |
| Resources | ≥ 20% host-RAM headroom at peak capture and restore; snapshot upload sustained below 50% of measured available bandwidth. |
| Cost | Snapshot storage, transfers, and extra capacity ≤ an agreed share of the worker's hourly cost, using dated prices ([Cloud Storage][gcs-pricing], [TPU][tpu-pricing]). |

**Go/no-go:** stop if runtime support or cross-VM restore fails. If pause, freshness, or cost gates fail, reduce snapshot size (Section 3.4) or lengthen the interval, and document the trade-off. Do not loosen gates after measurement to make an experiment pass.

## 3. Design

### 3.1 Process layout

```text
Spot TPU VM (tpu-v6e-8), host OS
├─ snapshot-agent (host process, outside the sandbox)
│    ├─ schedules checkpoints every I minutes
│    ├─ controls the admission firewall rule during capture
│    ├─ uploads completed images to GCS and publishes a manifest
│    └─ watches GCE metadata for the preemption notice
└─ runsc sandbox (Docker --runtime=runsc --tpuproxy, gVisor netstack on the docker bridge,
     │             -p 8000:8000, VBAR_CONTROL_SERVICE_URL=<docker0 IP>:8353)
     └─ vllm serve  (API server + EngineCore + TPU workers, all inside;
          │          VLLM_SERVER_DEV_MODE=1 for /pause and /resume)
          └─ 8 × TPU v6e chips via gVisor tpuproxy, libtpu ≥ 0.0.48 with checkpointing enabled

Durable, outside the VM:
  gs://<bucket>/snapshots/<generation>/   image files + manifest (written last)
  gs://<bucket>/hf-cache, xla-cache       weights and compilation cache, for cold-start fallback
Stable endpoint:
  DNS name (Cloud DNS) updated by the recovery step to the new VM's IP
```

There is no external gateway or journal. The only components outside the VM are storage, a DNS record, and SkyPilot's managed-job controller, which relaunches the VM.

The **entire** vLLM process tree must be inside one sandbox: the API server with its response store, EngineCore, and all TPU worker processes. Inventory the actual process tree and file descriptors in Phase 0 [TPU worker][repo-worker].

### 3.2 Taking a snapshot (and rejecting requests during it)

Every `I` minutes, the snapshot-agent:

1. **Closes admission.** It adds a host firewall rule that rejects *new* TCP connections to the vLLM port with a TCP reset, for example an `iptables` REJECT rule on `--syn` packets. New clients fail immediately and retry; these are category A requests. Already-established connections are left alone. This needs no change to vLLM.
2. **Reaches a safe point.** It calls `POST /pause?mode=keep&clear_cache=false` and confirms `GET /is_paused` returns `true`. vLLM finishes the current step and stops scheduling, while keeping every running and queued request and its KV cache. No custom engine hook is needed. Phase 0 showed both parameters matter: checkpointing a busy engine kills it, and omitting `clear_cache=false` crashes the TPU worker (Section 0). The CRIU TPU plugin's pause hook does not quiesce TPU work either [CRIU proposal][criu-tpu].
3. **Checkpoints.** It runs `runsc --root /var/run/docker/runtime-runc/moby --tpuproxy checkpoint --image-path=/var/lib/snapshots/<gen> --leave-running <container-id>` directly. Do **not** use `docker checkpoint`: it pauses the container first and deadlocks libtpu's control thread. gVisor signals libtpu (`ACTION_CHECKPOINT`); libtpu copies HBM into process memory and detaches the devices; runsc saves the memory image; with `--leave-running`, the sandbox then resumes and libtpu reattaches [runsc checkpoint][gvisor-checkpoint], [control][gvisor-control].
4. **Resumes serving.** It waits for libtpu's reattach acknowledgement, calls `POST /resume`, and removes the firewall rule.
5. **Uploads in the background.** It copies the immutable image directory to `gs://<bucket>/snapshots/<gen>/`, verifies checksums, and writes `manifest.json` **last**. Only generations with a manifest are eligible for restore. The previous generation is kept until the new one is published.

The serving pause covers steps 2–4, dominated by the HBM→host copy and memory dump. Requests already running are delayed during it, not dropped. With the small test program, a `--leave-running` checkpoint paused it for 42.6 s (7 GB image). For vLLM, Phase 0 measured only stop-checkpoints (155 s for a 92 GB image), so the `--leave-running` pause at that size still has to be measured in Phase 1. The sandbox **must use gVisor's netstack**: host-networking sockets cannot be saved [runsc networking][gvisor-checkpoint], and vLLM's internal PyTorch gloo socket aborted on restore under `--network=host`. Whether streaming client connections survive a `--leave-running` checkpoint under netstack is still untested. Background mode avoids depending on it.

**Manifest contents:** generation number, capture time, image file sizes and checksums, image digest, runsc version, libtpu/JAX versions, model and tokenizer revisions, TPU type/topology, vLLM flags, and the list of request IDs in the snapshot (for debugging and client support).

### 3.3 Preemption and restore

1. **Preemption notice** (snapshot-agent polls the GCE metadata `instance/preempted` value): close admission with the firewall rule. If a snapshot can finish before the VM is reclaimed, which is unlikely at this size, take one. Otherwise do nothing. Correctness relies on the last published generation.
2. **Relaunch:** SkyPilot's managed job (`sky jobs launch`, with recovery enabled) acquires a new v6e-8 in `asia-south1`, failing over between zones.
3. **Pick a snapshot:** the setup step finds the newest generation with a complete manifest, checks compatibility (same image digest, runsc, libtpu, topology, model revision), and downloads and verifies the files. If no compatible snapshot exists, it cold-starts vLLM from the weight and XLA caches. In that case all in-flight work is lost and every client retries.
4. **Restore:** create a container with an **identical** spec, then restore into it. Phase 0 used `docker create` followed by `docker start --checkpoint=<name>`, with the image under `/var/lib/docker/containers/<id>/checkpoints/<name>`. That worked, but took 756 s for 92 GB, because Docker copies the image into containerd's content store and then into a temporary directory; both locations needed RAM disks. Phase 1 should switch to `runsc restore --image-path=...` called directly. gVisor signals libtpu (`ACTION_RESTORE`); libtpu reopens the TPU devices, remapping device identities on the new host [device remapping][gvisor-remapping], and copies the data back into HBM (165 s in Phase 0).
5. **Health gate, then resume:** vLLM comes back still paused. Wait until gVisor logs libtpu's `ACTION_RESTORE` success, because gVisor resumes application threads before libtpu finishes, and TPU work in that window crashes the process. Then check `/health` and `GET /is_paused` (expect `true`), and call `POST /resume`.
6. **Switch the endpoint:** update the DNS record (low TTL, such as 30 s) to the new VM's IP, then open the firewall. Background-mode clients resume polling their response IDs. Clients of categories A–C retry against the same DNS name.

**Confirmed in Phase 0:** under netstack on the Docker bridge, the restored API server accepted connections on the new VM through the published port (`/health` 200), and background-mode requests kept running and stayed retrievable by ID. Client connections that were open at checkpoint time are not carried to the new VM, as expected.

### 3.4 Snapshot size and interval

| Quantity | Meaning |
| --- | --- |
| `H` | HBM bytes copied to host. Conservatively, the weights (~52 GB) plus all pre-allocated KV cache and other live buffers. |
| `S` | Uploaded image size: host process memory + `H` + runtime overhead, after any compression. |
| `P` | Serving pause per snapshot. |
| `I` | Interval between snapshots. |
| `B_up`, `B_down` | Sustained usable upload/download throughput under load. |

```text
upload time   >= S / B_up            (must be < I for a serial pipeline)
max rollback  ~= I + upload time     (age of the newest complete snapshot)
restore time  >= S / B_down + runsc restore + HBM rehydration + DNS switch
```

Illustrative arithmetic, not a measurement: a 200 GiB image at 1 GiB/s takes at least about 200 s to upload, so `I` must be well above roughly 4 minutes, and rollback after preemption could reach about 7–8 minutes of work.

**Phase 0 measurements** (Qwen3-30B-A3B, `--gpu-memory-utilization 0.4`):
- `S` ≈ 92–95 GB;
- stop-checkpoint 155 s;
- VM-to-VM copy at about 500 MB/s;
- restore 756 s through Docker, plus 165 s for libtpu to rehydrate HBM.

GCS upload/download throughput has not been measured. The VM's 97 GB boot disk cannot stage an image this size, even though SkyPilot was asked for `disk_size: 512`. Use host RAM (1.4 TB on v6e-8) or an attached data disk for staging.

Ways to shrink `S`, each needing measurement:
- **Lower vLLM's memory-utilization setting** so it pre-allocates less KV cache. Size it for the real concurrency of about 100 requests, not the maximum. This is the simplest, biggest lever.
- **Exclude zero pages:** runsc can skip zero-filled host pages. Whether libtpu's HBM-to-host copy leaves unused KV regions as zero pages is unverified. Don't count on it in the conservative estimate.
- **Compression:** trades CPU time and pause length for upload size.
- **Weights excluded from the image:** requires selective capture and reloading weights on restore. That is a separate engineering project and is out of scope for version 1 [KV deletion][repo-kv], [state refs][repo-state].

Host RAM must hold normal process memory, `H`, and capture overhead at the same time. Check the v6e-8 host machine type's memory against the measured peak.

### 3.5 Changes to `tpu-test.yaml` (Phase 2)

```yaml
# sky jobs launch -n gemma4-snap --secret HF_TOKEN tpu-test.yaml
resources:
  infra: gcp/asia-south1
  accelerators: tpu-v6e-8
  accelerator_args:
    runtime_version: v2-alpha-tpuv6e
  use_spot: true
  job_recovery: FAILOVER            # verify field names against the installed SkyPilot
  disk_size: 512                    # local staging for one image + headroom

file_mounts:
  /mnt/hf-cache:  {source: gs://<bucket>/hf-cache,  mode: MOUNT_CACHED}
  /mnt/xla-cache: {source: gs://<bucket>/xla-cache, mode: MOUNT_CACHED}

setup: |
  # Phase 0 setup that worked (see Section 0):
  # - docker + a pinned runsc build; `runsc install --runtime=runsc-tpu -- --tpuproxy`
  # - sudo mv /dev/vfio/devices aside (runsc bug); sysctl vm.max_map_count=4194304
  # - image = vllm/vllm-tpu:gemma4 + `pip install --no-deps libtpu==0.0.49` (pin the digest)
  # - large staging space (RAM disk or data disk) for snapshots

run: |
  # 1. find newest complete gs://<bucket>/snapshots/<gen>/manifest.json compatible with this VM
  # 2. if found: download, verify, `runsc restore`, wait for libtpu ACTION_RESTORE success,
  #    check /is_paused, POST /resume;
  #    else: docker run --runtime=runsc-tpu -p 8000:8000 ... vllm serve with
  #      LIBTPU_CHECKPOINTING_ENABLED=true LIBTPU_CHECKPOINTING_DMA_PIPELINE_ENABLE=false
  #      VBAR_CONTROL_SERVICE_URL=<docker0 IP>:8353 VLLM_SERVER_DEV_MODE=1
  #      VLLM_ENABLE_RESPONSES_API_STORE=1 HF_HOME/VLLM_XLA_CACHE_PATH on the mounts, --api-key
  # 3. health check + test inference
  # 4. update DNS record, open firewall
  # 5. start snapshot-agent (pause → runsc checkpoint → resume, upload, preemption watch)
```

`VLLM_SERVER_DEV_MODE=1` also exposes other development endpoints. Keep the port reachable only through the firewall, and require `--api-key`.

`runsc` is driven directly instead of through `docker checkpoint`. gVisor documents checkpoint/restore through raw `runsc` commands [runsc checkpoint][gvisor-checkpoint], and Phase 0 showed Docker's checkpoint path deadlocks libtpu. Docker's restore path works but is slow.

## 4. Implementation sequence and stop conditions

Effort ranges assume TPU capacity is available. They are not delivery commitments. An unsupported libtpu blocks everything after Phase 0.

| Phase | Effort / owner role | Deliverable and exit gate |
| --- | --- | --- |
| 0. Runtime probe | **Done** (October 6–7, 2026) | libtpu ≥ 0.0.48 supports the protocol. In-place, same-VM, and cross-VM restores passed, and one vLLM 100-request cross-VM trial passed. See Section 0. |
| 1. Sizing and baselines | 1–2 days; benchmark | Switch restore to direct `runsc restore`. Measure `--leave-running` pause with vLLM, GCS upload/download, host RAM peak, and cold/warm restart time. Test Gemma 4 with libtpu 0.0.49. Choose the memory-utilization setting and `I`. Freeze the gates. |
| 2. vLLM snapshot and restore | 3–4 days; engine + runtime | Snapshot-agent (firewall, `/pause` → checkpoint → `/resume`, upload, manifest), restore path with the libtpu-ack wait, health gate, DNS switch, SkyPilot managed job. Exit: the 100-request scenario passes on a different VM, including a different zone. |
| 3. Fault tests and decision | 2–3 days; benchmark + security | Run the failure matrix below with ≥ 30 trials and make an explicit go/no-go decision against Section 2. |

### Correctness experiment

1. Uninterrupted reference: submit 100 background requests (greedy, then seeded server-level sampling) and record all outputs.
2. Repeat, snapshot at step `k`, stop the source VM, restore on another VM.
3. Before resuming: compare scheduler queues, request token counts, selected **used** KV blocks, and the sampling RNG state (`rng_params_for_sampling` and fused-loop keys, not only the initial `rng_key` [Runner RNG][repo-rng], [sampling][repo-sampling]) against the source at step `k`.
4. Resume, then compare every request's final output with the reference.
5. Repeat at different snapshot points, occupancies, and context lengths, and in a different zone.

Bit-identical sampled continuation is claimed only for an unchanged execution order. New arrivals after restore change batch composition and can change later sampled tokens.

### Failure matrix

Preemption during capture; preemption during upload (previous generation must be used); missing or corrupt manifest; corrupt image files; incompatible image/runsc/libtpu/topology; insufficient host RAM at capture or restore; libtpu fails to reattach on the source after `--leave-running`; libtpu fails to rehydrate on the target; restored listener not reachable; DNS not updated; requests arriving during capture (must be refused fast); streaming clients during capture and at preemption; background request polled before restore completes (must get a retryable error, not "not found"); no compatible snapshot (cold start); no Spot capacity for an extended period.

## 5. Security, retention, and credentials

- Snapshots contain prompts, outputs, the response store, and in-memory secrets (`HF_TOKEN`, the vLLM API key). Treat them as sensitive: private bucket, TLS in transit, default or CMEK encryption at rest [Encryption][gcs-encryption], least-privilege IAM [IAM roles][gcs-iam]. The VM's service account gets create-only access for uploads and read access for restore; a separate janitor identity holds delete permission.
- Credentials captured in the image may be stale on the new VM. Prefer short-lived credentials fetched by the host-side agent rather than ones held inside the sandbox. Test that the restored server still works with the new VM's identity.
- Retention: keep the latest two complete generations, delete incomplete uploads after one hour, and make snapshots older than 24 h ineligible for restore. Add lifecycle rules as a backstop [Lifecycle management][gcs-lifecycle]. Soft-delete settings can extend physical retention.
- Start with synthetic prompts. Before real traffic, set retention from the service's data policy.

## 6. Risks and deferred work

Likelihood labels are hypotheses until measured.

| Risk | Likelihood | Impact | Mitigation / gate | Owner role |
| --- | --- | --- | --- | --- |
| Image's libtpu lacks the checkpoint protocol or cross-VM restore | **Resolved in Phase 0** for libtpu ≥ 0.0.48 | The shipped image (0.0.38) can't checkpoint | Pin a libtpu ≥ 0.0.48 image and re-qualify each libtpu/JAX/vLLM upgrade. | Runtime |
| Image size makes capture pauses, upload time, or rollback too large | **Measured: 92 GB, 155 s checkpoint, 756 s Docker restore** | Long pauses, stale snapshots, more retries | Lower KV pre-allocation; direct `runsc restore`; measure GCS; adjust `I`. | Infrastructure |
| Snapshot taken mid-TPU-step is inconsistent | **Observed: busy engine dies at checkpoint** | Lost server | `/pause?mode=keep&clear_cache=false` before every checkpoint; abort the checkpoint if `/is_paused` is not `true`. | Engine |
| Restore race: app threads run before libtpu reattaches | **Observed** | Crash right after restore | Keep vLLM paused until libtpu's `ACTION_RESTORE` success, then `/resume`. | Engine |
| gVisor/vLLM workarounds break on upgrade (`/dev/vfio/devices`, `max_map_count`, `clear_cache` bug, dev-mode endpoints) | Medium | Setup or checkpoint failures | Pin runsc, image, and libtpu versions; rerun the Phase 0 probe on every upgrade; report the upstream bugs. | Runtime |
| Streaming clients lose their stream at every preemption (and maybe at every snapshot) | Expected | Retries, wasted work | Background Responses API; client retry contract; test `--leave-running` socket behaviour. | Serving |
| Background Responses store unsupported or unbounded in the pinned image | Unknown | No way to retrieve restored results | Phase 0 check; delete retrieved responses; upgrade the image if needed. | Serving |
| Category B requests lost silently | Expected | Client confusion | Clients treat disconnects as retryable; documented API contract. | Serving |
| Restored endpoint not reachable (new IP, listener, DNS caching) | Medium | Longer outage | DNS with low TTL; listener test; health gate. | Infrastructure |
| Snapshot contains sensitive data or stale credentials | Expected | Exposure or failed auth | Section 5 controls. | Security |
| Compatible Spot capacity unavailable | Workload-dependent | Long outage despite a valid snapshot | Zone failover; record capacity wait separately. | Infrastructure |

**Deferred:** external request journal and gateway (would remove categories B and C but was explicitly excluded); excluding weights from snapshots; multimodal inputs; multi-host TPU slices; Ray executors; LoRA; speculative decoding.

## Appendix A. External technology findings

- **gVisor TPU support:** `tpuproxy` forwards TPU driver operations for single-VM TPU shapes [TPU support][gvisor-tpu]. On checkpoint, gVisor sends `ACTION_CHECKPOINT` to libtpu's control thread; on restore it sends `ACTION_RESTORE` and remaps device identities [control][gvisor-control], [protocol][gvisor-protocol], [remapping][gvisor-remapping].
- **libtpu (Phase 0):** releases 0.0.48 and 0.0.49 embed gVisor's `tpu_control.proto` (`cloud_gvisor.tpu.ControlAction`, `RuntimeState`), along with `LIBTPU_CHECKPOINTING_*` settings and a `vbar_control_service_url` / `VBAR_CONTROL_SERVICE_URL` option. Releases 0.0.38–0.0.47 do not.
- **CRIU TPU plugin (open PR):** documents libtpu advertising a control channel through a specially named thread, and moving HBM into host memory on checkpoint. It was build-tested without TPU hardware validation [CRIU proposal][criu-tpu].
- **llm-d RL time-slicing (open PR):** adds a TPU checkpoint backend using `tpucheckpoint` [llm-d TPU backend][llmd-tpu]. It suggests active upstream use of the protocol, but does not prove support in this image's libtpu.
- **Castform `feat/runsc-cr`:** a GPU-only, same-host runsc pattern, useful as a reference for safe-point adapters and packaging. It is not a cross-host TPU solution [comparison][castform-compare], [README][castform-readme], [restore checks][castform-engine].
- **GKE managed Pod snapshots** exclude TPUs [limitations][gke-snapshots]. **Multi-tier checkpointing** relies on Orbax [guide][mtc-docs]. **PJRT** has no live-client checkpoint API [overview][pjrt-overview], [C API][pjrt-api]. None replaces this design.

## Appendix B. Verified repository findings

| Finding | Implication | Source |
| --- | --- | --- |
| `TpuPlatform.validate_request` rejects per-request seeds ("JAX does not support per-request seed."). | Sampling randomness is engine-global, so preserving the engine's RNG state through the snapshot is the only way to keep sampled continuations identical. | [Seed check][repo-seed] |
| The runner seeds a global `rng_key` and advances sampling keys during generation. | The correctness experiment must compare the live RNG state, not only the initial seed. | [Runner RNG][repo-rng], [sampling][repo-sampling] |
| The worker sizes the KV cache from its HBM budget. | The image includes a large pre-allocated KV cache unless the memory setting is lowered. | [Worker memory sizing][repo-memory] |
| Worker sync only calls `jax.effects_barrier()`; `sleep()`/`wake_up()` raise `NotImplementedError`. | A safe-point hook must be added; there is no existing detach/reattach API. | [TPU worker][repo-worker] |
| `VLLM_XLA_CACHE_PATH` configures the persistent compilation cache. | Speeds up the cold-start fallback when no snapshot is usable. | [Compilation manager][repo-compilation] |
| Dependencies pin JAX/jaxlib `0.11.0` and libtpu `0.0.44`; no Orbax dependency. | **Phase 0: 0.0.44 has no checkpoint protocol.** The pin must move to ≥ 0.0.48 (tested with 0.0.49 + JAX 0.11.2). | [Requirements][repo-requirements] |
| vLLM 0.17.2 `EngineCoreProc.pause_scheduler` clears caches whenever `clear_cache` is true, even in `keep` mode, and the TPU worker doesn't implement `reset_encoder_cache`. | Always pass `clear_cache=false` with `mode=keep`. | `vllm/v1/engine/core.py` in the image (observed in Phase 0) |
| The runner holds KV caches, model state, RNG, and async bookkeeping as JAX state. | A full-process snapshot is required to preserve exact state without engine-level serialization. | [TPU runner][repo-runner], [state refs][repo-state] |

## References

Sources were reviewed during September 30–October 5, 2026. Mutable documentation and open proposals can change. Repository links pin the reviewed commit. Castform links require access to that private repository.

- Google Cloud: [Compute Engine Spot preemption][compute-spot], [GKE Spot shutdown][gke-spot], [GKE Pod snapshot limitations][gke-snapshots], [multi-tier checkpointing article][mtc-blog], [integration guide][mtc-docs], [Cloud Storage encryption][gcs-encryption], [IAM roles][gcs-iam], [lifecycle][gcs-lifecycle], [locations][gcs-locations], [pricing][gcs-pricing], [TPU pricing][tpu-pricing]
- gVisor / libtpu / CRIU: [TPU support][gvisor-tpu], [checkpoint/restore][gvisor-checkpoint], [TPU control][gvisor-control], [control protocol][gvisor-protocol], [device remapping][gvisor-remapping], [tpucheckpoint][tpucheckpoint], [CRIU TPU plugin][criu-tpu], [llm-d TPU checkpoint backend][llmd-tpu]
- PJRT, JAX, llm-d, Castform: [PJRT overview][pjrt-overview], [PJRT C API][pjrt-api], [JAX compilation cache][jax-cache], [llm-d][llmd], [Castform comparison][castform-compare], [README][castform-readme], [restore checks][castform-engine]
- Inference code: [seed check][repo-seed], [runner RNG][repo-rng], [sampling][repo-sampling], [requirements][repo-requirements], [runner state][repo-runner], [state refs][repo-state], [worker][repo-worker], [memory sizing][repo-memory], [weight updates][repo-weight-update], [compilation cache][repo-compilation], [KV deletion][repo-kv], [CPU backend][repo-cpu], [TPU connectors][repo-platform]

[gke-spot]: https://docs.cloud.google.com/kubernetes-engine/docs/concepts/spot-vms#termination-graceful-shutdown
[compute-spot]: https://docs.cloud.google.com/compute/docs/instances/spot#preemption_process
[gke-snapshots]: https://docs.cloud.google.com/kubernetes-engine/docs/concepts/pod-snapshots#limitations
[mtc-blog]: https://cloud.google.com/blog/products/ai-machine-learning/using-multi-tier-checkpointing-for-large-ai-training-jobs
[mtc-docs]: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing
[gvisor-tpu]: https://gvisor.dev/docs/user_guide/tpu/
[gvisor-checkpoint]: https://gvisor.dev/docs/user_guide/checkpoint_restore/
[gvisor-control]: https://github.com/google/gvisor/blob/aa5869c9615e99d579e886a2b7bf1ec6202f7cf3/pkg/sentry/control/tpu_control.go
[gvisor-protocol]: https://github.com/google/gvisor/blob/aa5869c9615e99d579e886a2b7bf1ec6202f7cf3/pkg/sentry/control/tpu_control.proto
[gvisor-remapping]: https://github.com/google/gvisor/blob/aa5869c9615e99d579e886a2b7bf1ec6202f7cf3/runsc/boot/tpuproxy.go
[tpucheckpoint]: https://github.com/google/gvisor/blob/aa5869c9615e99d579e886a2b7bf1ec6202f7cf3/tools/tpucheckpoint/tpucheckpoint.go
[criu-tpu]: https://github.com/checkpoint-restore/criu/pull/3128
[llmd-tpu]: https://github.com/llm-d-incubation/llm-d-rl-time-slicing/pull/194
[pjrt-overview]: https://openxla.org/xla/pjrt/cpp_api_overview
[pjrt-api]: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
[jax-cache]: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
[llmd]: https://github.com/llm-d/llm-d
[castform-compare]: https://github.com/castform-ai/castform/compare/main...feat/runsc-cr
[castform-readme]: https://github.com/castform-ai/castform/blob/d6196ef7d4978575a39ffeb95ef73333f956cc2c/core/runsc-cr/README.md
[castform-engine]: https://github.com/castform-ai/castform/blob/d6196ef7d4978575a39ffeb95ef73333f956cc2c/core/runsc-cr/runsc_cr/engine.py
[repo-seed]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/platforms/tpu_platform.py#L571
[repo-rng]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/tpu_runner.py#L864
[repo-requirements]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/requirements.txt#L8
[repo-runner]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/tpu_runner.py#L779
[repo-worker]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/worker/tpu_worker.py#L869
[repo-compilation]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/compilation_manager.py#L77
[repo-cpu]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/offload/cpu_backend.py#L26
[repo-platform]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/platforms/tpu_platform.py#L476
[repo-memory]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/worker/tpu_worker.py#L512
[repo-weight-update]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/worker/tpu_worker.py#L749
[repo-kv]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/kv_cache_manager.py#L1329
[repo-state]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/tpu_runner.py#L1303
[repo-sampling]: https://github.com/vllm-project/tpu-inference/blob/68e5a055b31808c246e95fc2967ffc12d9906ecb/tpu_inference/runner/tpu_runner.py#L2066
[gcs-encryption]: https://docs.cloud.google.com/storage/docs/encryption/default-keys
[gcs-iam]: https://docs.cloud.google.com/storage/docs/access-control/iam-roles
[gcs-lifecycle]: https://docs.cloud.google.com/storage/docs/lifecycle
[gcs-locations]: https://docs.cloud.google.com/storage/docs/locations
[gcs-pricing]: https://cloud.google.com/storage/pricing
[tpu-pricing]: https://cloud.google.com/tpu/pricing
