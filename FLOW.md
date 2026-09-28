# Execution Flow

Maintain the execution path as the code changes.

## Intended Flow

```text
python -m recurrent_memory.cli
  -> cli.main -> config.load_and_validate
  -> data.synthetic.build_splits -> data.loaders.make_loaders
  -> models factory (Transformer or TwoTimescale)
  -> train.fit
       -> model.forward_chunk
       -> loss.backward / optimizer step
       -> checkpoint save
  -> evaluate.evaluate_splits
  -> logging_utils.write_run_summary
```

## Current change path
*Phase 8 (Main Experiment):* Integrated dynamic configs and an automated runner (scripts/run_experiments.py) to execute and plot cross-seed evaluations.
## Automated Experiment Flow
`	ext
python scripts/run_experiments.py
  -> loops over [baseline, two_timescale] and seeds [17, 42, 100]
  -> invokes python -m recurrent_memory.cli smoke ...
  -> parses metrics.json (acc, memory, runtime)
  -> outputs xperiment_results.png
`
 
