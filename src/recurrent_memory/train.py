import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits = model(x)
        
        # Flatten logits and targets
        # logits: (B, seq_len, vocab_size)
        # y: (B, seq_len)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_last = 0
    total_last = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            
            # For the synthetic task, the actual query answer is at position seq_len - 2
            # because the input ends with [query_ind, query_key, answer_is_here, pad]
            # y has the answer at seq_len - 2.
            # Let's just check the exact match of the last non-pad target
            # Actually, y only has the answer at the specific query position. Let's just measure overall accuracy on valid tokens.
            preds = torch.argmax(logits, dim=-1)
            mask = (y != 0)
            correct_last += (preds[mask] == y[mask]).sum().item()
            total_last += mask.sum().item()
            
    return total_loss / len(dataloader), correct_last / total_last

def fit(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    
    best_val_acc = 0.0
    for epoch in range(config.train.epochs):
        print(f"Epoch {epoch+1}/{config.train.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return best_val_acc
