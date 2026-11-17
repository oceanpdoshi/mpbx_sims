"""JAX backend / precision setup for the TFLN mode-solve scripts.

Call ``setup_jax()`` BEFORE importing mpbjax (or tfln_rib / tfln_modes, which import it).
``import mpbjax`` initializes the JAX backend (mpbjax/helmholtz.py builds a module-level
jnp array), so a platform choice made after that import is silently ignored.

CPU vs GPU
----------
* This venv (native Windows, jax 0.12 CPU wheel) can only run on CPU: there is no
  CUDA plugin installed and JAX publishes CUDA wheels for Linux only.
* Single GPU later (Linux or WSL2): ``uv add "jax[cuda12]"`` (or ``pip install -e
  ../mpbx/python[cuda]``), then ``setup_jax("cuda")``. mpbx needs no other switch
  (docs/installation.md: "Nothing else changes from CPU. Keep JAX_ENABLE_X64=1").
  mpbx runs in float64; consumer GeForce cards have slow FP64, so benchmark first.
"""
import os
import sys


def setup_jax(platform="cpu", cache_dir=None, verbose=True):
    """Pin the JAX platform, enable the persistent compilation cache, check FP64.

    platform  : "cpu" (default) or "cuda". Use None to let JAX choose.
    cache_dir : on-disk XLA compilation cache, so a new process does not pay the
                compile cost again. Defaults to $JAX_COMPILATION_CACHE_DIR or
                ~/.cache/jax_mpbx. Pass False to disable it.
    Returns the imported ``mpbjax`` module.
    """
    if "mpbjax" in sys.modules and verbose:
        print("[tfln_setup] WARNING: mpbjax was imported before setup_jax(); "
              "the platform choice below may be ignored.")
    if platform is not None:
        os.environ["JAX_PLATFORMS"] = platform
    os.environ.setdefault("JAX_ENABLE_X64", "1")      # mpbjax also sets this on import

    import jax
    if platform is not None:
        jax.config.update("jax_platforms", platform)
    if cache_dir is not False:
        cache_dir = cache_dir or os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "jax_mpbx"))
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # the solver's per-eigensolve programs compile in < 1 s; cache them anyway
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    # mpbx error messages / HG labels contain non-ASCII characters (λ, ε, ₀); the
    # Windows cp1252 console would raise UnicodeEncodeError when printing them.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import mpbjax
    mpbjax.assert_fp64()                                 # raises unless float64 default
    if verbose:
        rep = mpbjax.precision_report()
        print(f"[setup_jax] backend={rep['backend']} devices={rep['devices']} "
              f"x64={rep['x64_enabled']} cache={cache_dir if cache_dir is not False else 'off'}")
    if platform is not None and jax.default_backend() != platform.replace("cuda", "gpu"):
        raise RuntimeError(f"requested platform {platform!r} but JAX is on {jax.default_backend()!r}")
    return mpbjax
