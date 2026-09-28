# Decision Log

Log meaningful implementation or research decisions here.

## 2026-09-29: Initial Research Contract
- **Context/question:** Narrowly operationalize the "Language Model Shape" idea for long-horizon agent trajectories.
- **Options considered:** Full LLM fine-tuning vs. minimal PyTorch prototype.
- **Decision:** Build a minimal synthetic PyTorch prototype predicting next-actions.
- **Rationale:** Focuses on the core systems/architectural tradeoff (recurrent vs dense) without the noise of LLM pretraining or massive compute costs. Fits Kaggle GPU constraints.
- **Tradeoff/risks accepted:** Synthetic tasks may not perfectly generalize to natural language agent trajectories.
- **Affected files/interfaces:** `README.md`, `ARCHITECTURE.md`, `src/recurrent_memory/models/`
- **Revisit when:** Moving to real language modeling or complex reinforcement learning environments.

## 2026-09-29: Dependency Management
- **Context/question:** How to specify dependencies.
- **Options considered:** `requirements.txt` vs `pyproject.toml`.
- **Decision:** Use `pyproject.toml` with `setuptools`.
- **Rationale:** Standard modern Python packaging, allows `pip install -e .` easily.
- **Tradeoff/risks accepted:** None.
- **Affected files/interfaces:** `pyproject.toml`
- **Revisit when:** Complex binary dependencies or pure Kaggle offline constraints mandate vendoring.

## 2026-09-29: Dynamic Configuration & Experiment Automation
- **Context/question:** Need to execute multi-seed experiments and compare resource costs (memory/time) without manual intervention.
- **Decision:** Upgraded configs to accept seq_len/
um_train, injected resource tracking into cli.py, and added scripts/run_experiments.py for orchestration.
- **Rationale:** A centralized python script handles the Kaggle execution loop cleanly, logs metrics, and outputs comparison plots immediately.
- **Affected files/interfaces:** config.py, loaders.py, cli.py, configs/*_main.yaml, scripts/run_experiments.py
