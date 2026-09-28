import argparse
import os
import json
import torch
from .config import load_config
from .data.loaders import make_loaders
from .models import get_model
from .train import fit, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Recurrent Memory Kaggle Project")
    subparsers = parser.add_subparsers(dest="command")
    
    smoke_parser = subparsers.add_parser("smoke", help="Run a smoke test")
    smoke_parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="test")
    eval_parser.add_argument("--output", type=str, required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == "smoke":
        config = load_config(args.config)
        config.train.seed = args.seed
        
        os.makedirs(args.output, exist_ok=True)
        
        # Save config
        # For simplicity, we just save the path
        with open(os.path.join(args.output, "config.yaml"), "w") as f:
            f.write(f"original_config: {args.config}\nseed: {args.seed}\n")
            
        torch.manual_seed(config.train.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        import time
        start_time = time.time()
        
        train_loader, val_loader, test_loader = make_loaders(config, vocab_size=config.model.vocab_size)
        
        model = get_model(config).to(device)
        
        # Train
        fit(model, train_loader, val_loader, config, device)
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'config_path': args.config,
            'seed': args.seed
        }, os.path.join(args.output, "checkpoint.pt"))
        
        # Eval test
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        end_time = time.time()
        
        peak_mem = 0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "runtime_sec": end_time - start_time,
            "peak_memory_mb": peak_mem
        }
        with open(os.path.join(args.output, "metrics.json"), "w") as f:
            json.dump(metrics, f)
            
        print(f"Smoke test complete. Metrics saved to {args.output}/metrics.json")
        print(metrics)

    elif args.command == "evaluate":
        ckpt = torch.load(args.checkpoint)
        config = load_config(ckpt['config_path'])
        config.train.seed = ckpt['seed']
        
        torch.manual_seed(config.train.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = get_model(config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        
        _, _, test_loader = make_loaders(config, vocab_size=config.model.vocab_size)
        
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f)
            
        print(f"Evaluation complete. Metrics saved to {args.output}")
        print(metrics)

if __name__ == "__main__":
    main()
