"""Minimal TPU state probe for checkpoint/restore tests.

Fills HBM on every chip with known data and keeps verifying it on device, so the
check reads HBM rather than a cached host copy. While /probe/PAUSE exists the
loop issues no TPU work, which simulates an engine safe point.
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

host = np.random.default_rng(0).standard_normal((4096, 4096), dtype=np.float32)  # 64 MiB per chip
u = host.view(np.uint32)
ref = (int(u.sum(dtype=np.uint32)), int(np.bitwise_xor.reduce(u, axis=None)))
devs = jax.devices()
arrs = [jax.device_put(host, d) for d in devs]


@jax.jit
def check(a):
    b = jax.lax.bitcast_convert_type(a, jnp.uint32)
    return (jnp.sum(b, dtype=jnp.uint32),
            jax.lax.reduce(b, np.uint32(0), jax.lax.bitwise_xor, (0, 1)),
            jnp.sum(a @ a))


print(f"devices={len(devs)} kind={devs[0].device_kind} ref={ref}", flush=True)
step = 0
while True:
    while os.path.exists("/probe/PAUSE"):  # safe point: no TPU work while detached
        time.sleep(0.2)
    res = [check(a) for a in arrs]
    ok = all((int(s), int(x)) == ref for s, x, _ in res)
    print(f"step={step} t={time.strftime('%X')} hbm_match={ok} matmul0={float(res[0][2]):.3f}", flush=True)
    step += 1
    time.sleep(5)
