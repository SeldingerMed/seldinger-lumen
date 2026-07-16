# Common Endovascular Benchmark

This directory holds the reproducible comparison harness for Lumen, CathSim, and
stEVE/SOFA-style environments. The contract is in `metric_contract.json`.

The benchmark compares matched task classes, not identical geometry. Pilot runs use
random, forward, and sweep policies to verify adapters and metric extraction. Main
paper-facing runs use PPO and SAC with CathSim-style budgets: 600,000 training steps,
six seeds, and 100 frozen-policy evaluation episodes per seed/task.

Example pilot commands:

```bash
python benchmarks/external_comparison/common_bench.py lumen \
  --episodes 30 \
  --policies random,forward,sweep \
  --run-id lumen-pilot

MUJOCO_GL=disable /path/to/cathsim-venv/bin/python \
  benchmarks/external_comparison/common_bench.py cathsim \
  --episodes 30 \
  --policies random,forward,sweep \
  --external-repo /path/to/cathsim \
  --run-id cathsim-pilot
```

stEVE/SOFA readiness is checked separately because stEVE requires SOFA, SofaPython3,
and BeamAdapter:

```bash
python benchmarks/external_comparison/common_bench.py smoke-steve --run-id steve-smoke
```

Combine aggregate rows after runs finish:

```bash
python benchmarks/external_comparison/summarize_results.py \
  benchmarks/external_comparison/results/lumen-pilot-30.json \
  benchmarks/external_comparison/results/cathsim-pilot-30.json
```
