import pytest
from recurrent_memory.data.synthetic import build_splits
import torch

def test_data_determinism():
    t1, v1, te1 = build_splits(num_train=10, num_val=5, seq_len=32, vocab_size=50, seed=42)
    t2, v2, te2 = build_splits(num_train=10, num_val=5, seq_len=32, vocab_size=50, seed=42)
    
    # Check that datasets are identical for same seed
    for (x1, y1), (x2, y2) in zip(t1, t2):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    # Check different seed produces different datasets
    t3, _, _ = build_splits(num_train=10, num_val=5, seq_len=32, vocab_size=50, seed=43)
    same_count = sum([torch.equal(x1, x3) for (x1, _), (x3, _) in zip(t1, t3)])
    assert same_count < len(t1), "Different seeds should produce different datasets"
