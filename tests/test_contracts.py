import torch
from recurrent_memory.data.synthetic import build_splits

def test_target_isolation():
    t, v, te = build_splits(num_train=10, num_val=5, seq_len=32, vocab_size=50, seed=42)
    x, y = t[0]
    # Ensure there is exactly 1 valid target token (not 0) in y
    assert (y != 0).sum().item() == 1
