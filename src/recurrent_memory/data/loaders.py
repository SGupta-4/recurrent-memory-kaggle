from torch.utils.data import DataLoader
from .synthetic import build_splits

def make_loaders(config, seq_len=128, vocab_size=100):
    train_ds, val_ds, test_ds = build_splits(
        num_train=1000, num_val=200, seq_len=seq_len, vocab_size=vocab_size, seed=config.train.seed
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
