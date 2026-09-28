# Bug Trace

Record bugs, feature work, attempts, and verification.

## 001: Initial Setup
- **Discovery/Scope:** Need to scaffold the project structure as defined in the plan.
- **Attempted Fixes:** Generated skeleton using scripts.
- **Selected Fix:** Adopted standard Python package structure under `src/`.
- **Files Changed:** Created directory tree and markdown documentation.
- **Verification:** Directories and files exist.
- **Status:** Done.

## 002: Main Experiment Automation
- **Discovery/Scope:** Need to scale up from the smoke test to the main configuration, varying sequence length and tracking peak GPU memory and runtime.
- **Selected Fix:** Parameterized seq_len in config.py, measured 	orch.cuda.max_memory_allocated() in cli.py, and added 
un_experiments.py to plot outputs.
- **Files Changed:** config.py, loaders.py, cli.py, configs/, scripts/run_experiments.py.
- **Status:** Done.
