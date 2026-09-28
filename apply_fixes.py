import os
import re

# 1. Fix synthetic.py Target Coping
with open('src/recurrent_memory/data/synthetic.py', 'r') as f:
    data = f.read()
data = re.sub(
    r"target = seq\.copy\(\).*?seq\[query_pos \+ 2\] = 0",
    "target = [0] * seq_len\n            target[query_pos + 1] = needle_val\n            seq[query_pos + 2] = 0",
    data, flags=re.DOTALL
)
with open('src/recurrent_memory/data/synthetic.py', 'w') as f:
    f.write(data)
    
# 2. Fix train.py Tracking Best Checkpoint
with open('src/recurrent_memory/train.py', 'r') as f:
    data = f.read()
data = data.replace('optimizer.zero_grad()', 'optimizer.zero_grad(set_to_none=True)')
data = data.replace('best_val_acc = 0.0\n    for epoch', 'best_val_acc = 0.0\n    best_state = None\n    import copy\n    for epoch')
data = data.replace('best_val_acc = val_acc', 'best_val_acc = val_acc\n            best_state = copy.deepcopy(model.state_dict())')
data = data.replace('return best_val_acc', 'if best_state is not None:\n        model.load_state_dict(best_state)\n    return best_val_acc')
with open('src/recurrent_memory/train.py', 'w') as f:
    f.write(data)
    
# 3. Fix cli.py Commands, Syncs, Evaluation Splitting
with open('src/recurrent_memory/cli.py', 'r') as f:
    data = f.read()
data = data.replace('add_parser("smoke"', 'add_parser("train"')
data = data.replace('smoke_parser', 'train_parser')
data = data.replace('args.command == "smoke"', 'args.command == "train"')
data = data.replace('start_time = time.time()', 'if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()\n        start_time = time.time()')
data = data.replace('end_time = time.time()', 'if torch.cuda.is_available(): torch.cuda.synchronize()\n        end_time = time.time()')
data = data.replace('peak_mem = 0', 'peak_mem = 0\n        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)')
data = data.replace('"peak_memory_mb": peak_mem', '"peak_memory_mb": peak_mem,\n            "trainable_parameters": trainable_params')
data = data.replace('Smoke test complete.', 'Train complete.')

# Fix the dataloader loading for evaluate 
data = data.replace('_, _, test_loader = make_loaders(config, vocab_size=config.model.vocab_size)', 'train_loader, val_loader, test_loader = make_loaders(config, vocab_size=config.model.vocab_size)\n        eval_loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(args.split, test_loader)')
data = data.replace('evaluate(model, test_loader, device)', 'evaluate(model, eval_loader, device)')
data = data.replace('ckpt = torch.load(args.checkpoint)', 'ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)')

with open('src/recurrent_memory/cli.py', 'w') as f:
    f.write(data)
    
# 4. Fix Markdowns
def fix_md(p, rs):
    if not os.path.exists(p): return
    with open(p, 'r') as f: c = f.read()
    for o, n in rs: c = c.replace(o, n)
    with open(p, 'w') as f: f.write(c)

fix_md('README.md', [('recurrent_memory.cli smoke', 'recurrent_memory.cli train'), ('notebooks/kaggle_run.ipynb', 'kaggle_run.ipynb')])
fix_md('FLOW.md', [('config.load_and_validate', 'config.load_config'), ('evaluate.evaluate_splits', 'train.evaluate'), ('logging_utils.write_run_summary', 'json.dump'), ('recurrent_memory.cli smoke', 'recurrent_memory.cli train')])
fix_md('ARCHITECTURE.md', [('`smoke`, `train`, `evaluate`', '`train`, `evaluate`'), ('recurrent vector', 'recurrent memory slots')])

with open('DECISION.md', 'a') as f:
    f.write('\\n## 2026-09-29: Objective and Metric Fix\\n- **Context:** Training was evaluating accuracy across the entire sequence of copied tokens.\\n- **Decision:** Shifted target creation to zero out all tokens except the actual next-token answer at the query position.\\n- **Rationale:** Ensures accuracy exclusively represents the model\\\'s ability to retrieve the target needle, eliminating trivial copying.\\n')

# 5. Add specific tests
with open('tests/test_contracts.py', 'w') as f:
    f.write('''import torch
from recurrent_memory.data.synthetic import build_splits

def test_target_isolation():
    t, v, te = build_splits(num_train=10, num_val=5, seq_len=32, vocab_size=50, seed=42)
    x, y = t[0]
    # Ensure there is exactly 1 valid target token (not 0) in y
    assert (y != 0).sum().item() == 1
''')
