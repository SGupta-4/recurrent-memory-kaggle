import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        # x shape: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[offset:offset+seq_len, :].unsqueeze(0)

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, seq_len)
        seq_len = x.size(1)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        
        # Causal mask for PyTorch Transformer (seq_len, seq_len)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.fc_out(out)
        return logits
