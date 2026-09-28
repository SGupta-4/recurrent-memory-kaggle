import torch
from recurrent_memory.models.transformer import BaselineTransformer
from recurrent_memory.models.recurrent_memory import TwoTimescaleModel

def test_baseline_shape():
    model = BaselineTransformer(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    x = torch.randint(0, 100, (4, 32))
    out = model(x)
    assert out.shape == (4, 32, 100)

def test_two_timescale_shape():
    model = TwoTimescaleModel(vocab_size=100, d_model=32, n_heads=2, n_layers=1, chunk_length=16, memory_slots=4)
    x = torch.randint(0, 100, (4, 32)) # 2 chunks
    out = model(x)
    assert out.shape == (4, 32, 100)
    
def test_two_timescale_memory_carry():
    model = TwoTimescaleModel(vocab_size=100, d_model=32, n_heads=2, n_layers=1, chunk_length=16, memory_slots=4)
    x = torch.randint(0, 100, (2, 16))
    
    # Initialize state
    mem = model.init_memory(2, x.device)
    assert mem.shape == (2, 4, 32)
    
    out, new_mem = model.forward_chunk(x, mem, chunk_offset=0)
    
    assert out.shape == (2, 16, 100)
    assert new_mem.shape == (2, 4, 32)
    # Check that memory updated
    assert not torch.allclose(mem, new_mem)
