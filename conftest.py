# Presence of this file makes pytest add the repo root to sys.path, so the
# top-level `benchmarks` package imports under a bare `pytest` invocation (CI),
# not only under `python -m pytest` (which adds the CWD itself).
