from torch.utils.data import DataLoader
from .synthetic import build_splits

def make_loaders(config, vocab_size=100):
    train_ds, val_ds, test_ds = build_splits(
        num_train=config.train.num_train, 
        num_val=config.train.num_val, 
        seq_len=config.train.seq_len, 
        vocab_size=vocab_size, 
        seed=config.train.seed
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
