#!/bin/bash
# Stage 4 (target VM): poll until all requests finish, then compare with the
# state recorded at snapshot time.
set -u
HERE="$(dirname "$0")"

for _ in $(seq 1 120); do
  python3 "$HERE/scen.py" report /mnt/snap/final.json | tee /tmp/rep.txt
  grep -qE "'queued'|'in_progress'" /tmp/rep.txt || break
  sleep 10
done
python3 - <<'P'
import json
s = json.load(open("/mnt/snap/at_snapshot.json")); f = json.load(open("/mnt/snap/final.json"))
done_before = [k for k, v in s.items() if v[0] in ("completed", "incomplete")]
same = sum(1 for k in done_before if f[k][2] == s[k][2])
print(f"finished before snapshot: {len(done_before)}; identical text after restore (not rerun): {same}")
frozen = [k for k in s if k not in done_before]
fin = sum(1 for k in frozen if f[k][0] in ("completed", "incomplete"))
print(f"frozen at snapshot: {len(frozen)}; finished after restore: {fin}")
print("final states:", {st: sum(1 for v in f.values() if v[0] == st) for st in set(v[0] for v in f.values())})
P
# Continuation check: prompt throughput should stay 0 after /resume (no re-prefill).
sudo docker logs vllm-gv 2>&1 | grep -oE 'Avg prompt throughput: [0-9.]+ tokens/s, Avg generation throughput: [0-9.]+ tokens/s, Running: [0-9]+ reqs' | tail -4
