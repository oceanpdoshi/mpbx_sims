"""CPU scaling benchmark: how many cores should ONE mode-solve process get?

    python bench_cpu.py                        all configurations on every CPU this process may use
    python bench_cpu.py --cores 0-23           restrict to these logical CPUs (e.g. one per physical core)
    python bench_cpu.py --iters 40             LOBPCG iterations timed per run (default 40)

Run it on an otherwise idle machine (on the cluster: inside an exclusive allocation of one node,
or at least of the full task, e.g. `srun --exclusive -c 48 python bench_cpu.py`).

Kernel: a fixed number of LOBPCG iterations (tol = 0, never stops early) of the jit-compiled
eigensolve that tfln_modes / run_chunked use, on chunk B's production grid (4.39 x 3.2 µm,
dx = dy = 12.5 nm -> 351 x 256 px, odd Nx), one x-parity sector, nev = 5, λ = 0.8 µm. A full
solve is ~(bands x Newton steps x iterations) of exactly this, so time/iteration is what matters.

Two questions are answered:
  * scaling: 1 process on 1, 2, 4, 8, ... cores -> speed-up and parallel efficiency
  * packing: N/c processes x c cores at the same time -> total throughput (iterations/s/node)
Each run is a fresh process pinned to its cores BEFORE jax is imported (os.sched_setaffinity on
Linux, psutil elsewhere); on Linux XLA sizes its thread pool from that CPU mask.
On the Windows i7-12700 desktop used for the README numbers, logical CPUs 0-15 are 8
hyper-threaded P-cores and 16-19 are E-cores; the default there is one logical CPU per P-core.
Results -> results/bench_cpu.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def pin(cores):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    else:
        import psutil
        psutil.Process().cpu_affinity(list(cores))


def worker(cores, iters):
    pin(cores)
    sys.path[:0] = [ROOT]
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)
    import jax
    import jax.numpy as jnp
    from mpbjax import grid_2d
    from geometry_gen import tfln_rib as R
    import sim_core.tfln_modes as M

    small = os.environ.get("BENCH_SMOKE")
    Nx, Ny, d = (91, 64, 0.05) if small else (351, 256, 0.0125)
    grid = grid_2d(Nx * d, Ny * d, Nx, Ny)
    geom = R.align_to_grid(R.RibGeometry(), grid, frac=1 / 3, warn=False)
    t0 = time.perf_counter()
    diel = R.RibDielectric(geom, grid)
    t_plan = time.perf_counter() - t0
    dd = diel.at(0.8)
    einv = jnp.asarray(dd["einv"])
    k = jnp.asarray(2.03 / 0.8)
    key = jax.random.PRNGKey(0)
    kw = dict(grid=grid, nev=5, tol=0.0, maxiter=iters, parity=-1)
    jax.block_until_ready(M._eigensolve(k, einv, None, key, **kw))       # compile + warm-up
    t0, c0 = time.perf_counter(), time.process_time()
    out = M._eigensolve(k, einv, None, key, **kw)                         # identical work, timed
    jax.block_until_ready(out)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)
        if peak is None:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:                                                      # noqa: BLE001
        peak = float("nan")
    print(json.dumps(dict(cores=cores, iters=int(out[2]), wall=wall, s_per_iter=wall / max(int(out[2]), 1),
                          cpu=cpu, peak_mem_gb=peak / 1e9, t_plan=t_plan)), flush=True)


def run_group(groups, iters):
    """Start one worker per core group simultaneously; return their JSON results."""
    procs = [subprocess.Popen([PY, __file__, "--worker", ",".join(map(str, g)), "--iters", str(iters)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for g in groups]
    res = []
    for pr in procs:
        out, err = pr.communicate()
        line = [ln for ln in out.splitlines() if ln.startswith("{")]
        res.append(json.loads(line[-1]) if line else dict(error=(err or out)[-800:]))
    return res


def parse_cores(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def default_cores():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    import psutil
    if psutil.cpu_count() == 20 and psutil.cpu_count(logical=False) == 12:   # i7-12700 (8P+4E)
        return list(range(0, 16, 2))
    return list(range(psutil.cpu_count(logical=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", default=None)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--worker", default=None)
    a = ap.parse_args()
    if a.worker is not None:
        worker([int(c) for c in a.worker.split(",") if c], a.iters)
        return
    sys.stdout.reconfigure(encoding="utf-8")
    P = parse_cores(a.cores) if a.cores else default_cores()
    N = len(P)
    sizes = [c for c in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if c <= N]
    if N not in sizes:
        sizes.append(N)
    configs = [(f"1 proc x {c} cores", [P[:c]]) for c in sizes]
    configs += [(f"{N // c} procs x {c} cores", [P[i * c:(i + 1) * c] for i in range(N // c)])
                for c in sizes if 1 <= c < N and N // c >= 2]
    print(f"[bench] {N} CPUs: {P}")
    out, base = [], None
    for name, groups in configs:
        res = run_group(groups, a.iters)
        ok = [r for r in res if "wall" in r]
        if not ok:
            print(f"{name}: FAILED {res[0].get('error', '')[:300]}")
            continue
        thr = sum(r["iters"] / r["wall"] for r in ok)                # LOBPCG iterations/s, all processes
        spi = sum(r["s_per_iter"] for r in ok) / len(ok)
        if base is None:
            base = spi                                                 # 1 proc x 1 core
        rec = dict(config=name, n_procs=len(groups), cores_per_proc=len(groups[0]), results=res,
                   throughput_iter_per_s=thr, mean_s_per_iter=spi,
                   speedup_vs_1core=base / spi, efficiency=base / spi / len(groups[0]),
                   node_throughput_vs_1core=thr * base)
        out.append(rec)
        print(f"{name:22s} {spi:7.3f} s/iter per process   speed-up {base / spi:5.2f} "
              f"(efficiency {100 * base / spi / len(groups[0]):3.0f} %)   all processes: {thr * base:5.2f}x one core   "
              f"cpu/wall {ok[0]['cpu'] / ok[0]['wall']:.1f}   peak mem {ok[0]['peak_mem_gb']:.2f} GB", flush=True)
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "bench_cpu.json"), "w") as f:
            json.dump(dict(cores=P, iters=a.iters, configs=out), f, indent=1)


if __name__ == "__main__":
    main()
