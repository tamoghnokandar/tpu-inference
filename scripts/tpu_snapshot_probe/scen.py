"""Client for the 100-request snapshot scenario (vLLM Responses API, background mode).

  python3 scen.py submit [N_DONE]   submit 100 requests, return once N_DONE have finished
  python3 scen.py report [OUT]      print status counts; optionally save {id: [status, chars, text]}
"""
import collections
import json
import sys
import time
import urllib.request

URL = "http://localhost:8000"
M = "Qwen/Qwen3-30B-A3B"
IDS = "/mnt/snap/ids.json"
TOPICS = ["TPUs", "spot VMs", "KV caches", "gVisor", "checkpointing", "paged attention",
          "MoE models", "XLA", "HBM", "schedulers"]


def req(path, body=None):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))


def text(r):
    return "".join(c.get("text", "") for o in r.get("output", []) for c in (o.get("content") or []))


def status_all(ids):
    out = {}
    for i in ids:
        try:
            r = req(f"/v1/responses/{i}")
            out[i] = (r.get("status"), len(text(r)), r)
        except Exception as e:
            out[i] = (f"ERR:{type(e).__name__}", 0, None)
    return out


cmd = sys.argv[1]
if cmd == "submit":
    ids = []
    for n in range(100):
        body = {"model": M, "background": True, "temperature": 0,
                "max_output_tokens": 300 + (n % 10) * 150,
                "input": f"Request {n}: write a detailed technical explanation of {TOPICS[n % 10]}. /no_think"}
        ids.append(req("/v1/responses", body)["id"])
    json.dump(ids, open(IDS, "w"))
    print(f"submitted {len(ids)}")
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    while True:
        st = status_all(ids)
        c = collections.Counter(s for s, _, _ in st.values())
        if c.get("completed", 0) + c.get("incomplete", 0) >= target:
            break
        time.sleep(1)
    print("READY_TO_SNAPSHOT", dict(c))
elif cmd == "report":
    ids = json.load(open(IDS))
    st = status_all(ids)
    print(time.strftime("%X"), dict(collections.Counter(s for s, _, _ in st.values())))
    if len(sys.argv) > 2:
        json.dump({i: [s, n, text(r) if r else ""] for i, (s, n, r) in st.items()}, open(sys.argv[2], "w"))
