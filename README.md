# Recurrent Memory for Long-Horizon Trajectories

This is a PyTorch research prototype testing a "two-timescale" model shape for long-horizon agent trajectories. It compares a standard dense causal Transformer (recent window / flat history) against a model that processes recent history densely but compresses older history into a recurrently updated compact memory state.

## Quick Start

1. Install dependencies:
```bash
python -m pip install -e .
```

2. Run a smoke test:
```bash
python -m recurrent_memory.cli train --config configs/smoke.yaml --seed 17 --output outputs/smoke-seed17
```

3. Evaluate the smoke test:
```bash
python -m recurrent_memory.cli evaluate --checkpoint outputs/smoke-seed17/checkpoint.pt --split test --output outputs/smoke-seed17/eval.json
```

## Full Experiment

To run the main comparison (baseline vs. two-timescale) across multiple seeds and generate a performance/memory plot:
`ash
python scripts/run_experiments.py
`

## Structure
- `src/recurrent_memory/`: Core implementation.
- `configs/`: YAML configurations for experiments.
- `notebooks/`: Kaggle notebook entrypoints.
- `tests/`: Verification and correctness checks.

For architectural details, see `ARCHITECTURE.md`. For logs and bug traces, see `BUGS.md`.
