# Architecture

High-level architecture of the Recurrent Memory experiment.

## Core System Boundary
The primary objective predicts the next action/decision from an instruction, prior actions/observations, and a current observation.

- **Baseline A (Recent-window/Full-history):** Densely attends over a flat bounded or full token history using a standard causal Transformer.
- **Treatment (Two-Timescale Trajectory Model):** Processes the recent context densely using a causal Transformer, while older trajectory is processed recurrently into a bounded, fixed-size Memory State.

## Data Flow
`Synthetic Generators -> Encoding -> Dataloaders -> Model (Chunks) -> Loss -> Checkpoint`

The core innovation resides in how the `forward` pass chunks the input trajectory in the two-timescale model, passing the dense window as standard attention queries and resolving the older history as an updated recurrent memory slots.

## Interfaces
- **Models:** Expose a unified `forward` interface taking chunks.
- **Data:** Deterministic synthetic task splits.
- **CLI:** Subcommands for `train`, `evaluate`.
