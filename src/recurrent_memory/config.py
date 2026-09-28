import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    type: str = "baseline" # "baseline" or "two_timescale"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    chunk_length: int = 64
    memory_slots: int = 16
    vocab_size: int = 1000

@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 1
    max_steps: Optional[int] = None
    seed: int = 42

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    model_conf = ModelConfig(**data.get("model", {}))
    train_conf = TrainConfig(**data.get("train", {}))
    return Config(model=model_conf, train=train_conf)
