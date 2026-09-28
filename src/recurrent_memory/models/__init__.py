from .transformer import BaselineTransformer
from .recurrent_memory import TwoTimescaleModel

def get_model(config):
    if config.model.type == "baseline":
        return BaselineTransformer(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers
        )
    elif config.model.type == "two_timescale":
        return TwoTimescaleModel(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            chunk_length=config.model.chunk_length,
            memory_slots=config.model.memory_slots
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
