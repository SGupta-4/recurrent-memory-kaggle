import torch
from torch.utils.data import Dataset
import numpy as np

class NeedleDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=100, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        
        # Token meanings:
        # 0: pad
        # 1: key indicator
        # 2: query indicator
        self.special_tokens = 3
        
        self.data = []
        for _ in range(num_samples):
            # Generate random distractors
            seq = self.rng.randint(self.special_tokens, self.vocab_size, size=seq_len).tolist()
            
            # Insert needle at a random early position
            needle_pos = self.rng.randint(0, seq_len // 2)
            needle_key = self.rng.randint(self.special_tokens, self.vocab_size)
            needle_val = self.rng.randint(self.special_tokens, self.vocab_size)
            
            seq[needle_pos] = 1 # key indicator
            seq[needle_pos + 1] = needle_key
            seq[needle_pos + 2] = needle_val
            
            # The query is at the end
            query_pos = seq_len - 3
            seq[query_pos] = 2 # query indicator
            seq[query_pos + 1] = needle_key
            
            # The target is the value we want to predict at the last position
            # We will formulate it as next-token prediction, so the target for query_pos+1 is needle_val
            target = [0] * seq_len
            target[query_pos + 1] = needle_val
            seq[query_pos + 2] = 0
            
            self.data.append((torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def build_splits(num_train=1000, num_val=100, seq_len=128, vocab_size=100, seed=42):
    train_dataset = NeedleDataset(num_train, seq_len, vocab_size, seed)
    val_dataset = NeedleDataset(num_val, seq_len, vocab_size, seed + 1)
    test_dataset = NeedleDataset(num_val, seq_len, vocab_size, seed + 2)
    return train_dataset, val_dataset, test_dataset
