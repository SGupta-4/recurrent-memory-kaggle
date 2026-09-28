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
*Initial setup phase:* Establishing the directory skeleton, core documentation, and the basic CLI execution hooks. 
