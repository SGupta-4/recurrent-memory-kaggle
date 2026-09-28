import torch
import torch.nn as nn
from .transformer import PositionalEncoding

class TwoTimescaleModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, chunk_length=32, memory_slots=16):
        super().__init__()
        self.chunk_length = chunk_length
        self.memory_slots = memory_slots
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # We need a custom block to attend to (Memory + Current Window)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Recurrent update for memory (simple GRU-like or linear)
        # We'll use a simple cross-attention from memory slots to the old chunk, then a gate
        self.memory_query = nn.Parameter(torch.randn(1, memory_slots, d_model))
        self.memory_update_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.memory_transform = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def init_memory(self, batch_size, device):
        # Initial memory is just the learned query expanded
        return self.memory_query.expand(batch_size, -1, -1).to(device)

    def forward_chunk(self, x, memory_state, chunk_offset):
        # x: (B, chunk_len)
        B, seq_len = x.size()
        
        emb = self.embedding(x)
        emb = self.pos_encoder(emb, offset=chunk_offset)
        
        # Causal mask for the current window
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Process through layers
        # The window items act as target, Memory acts as extra past context
        # In a standard TransformerDecoderLayer: tgt is self-attended (causal), then cross-attends to memory.
        # This matches our requirement: recent window is dense causal, old history is memory.
        curr_state = emb
        for layer in self.layers:
            curr_state = layer(tgt=curr_state, memory=memory_state, tgt_mask=tgt_mask)
            
        logits = self.fc_out(curr_state)
        
        # Update memory state using the just-processed chunk
        # memory_state cross-attends to the curr_state (which encapsulates the chunk)
        mem_update, _ = self.memory_update_attn(query=memory_state, key=curr_state, value=curr_state)
        
        # Gate the update
        gate = self.memory_gate(torch.cat([memory_state, mem_update], dim=-1))
        new_memory_state = memory_state * (1 - gate) + self.memory_transform(mem_update) * gate
        
        return logits, new_memory_state

    def forward(self, x):
        # Full sequence forward pass (for training/evaluation loop compatibility)
        B, seq_len = x.size()
        device = x.device
        
        memory_state = self.init_memory(B, device)
        all_logits = []
        
        for i in range(0, seq_len, self.chunk_length):
            chunk = x[:, i:i+self.chunk_length]
            logits, memory_state = self.forward_chunk(chunk, memory_state, chunk_offset=i)
            all_logits.append(logits)
            
        return torch.cat(all_logits, dim=1)
