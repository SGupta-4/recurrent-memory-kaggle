import os
import json
import sys

def main(output_dir):
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics.json found in {output_dir}")
        return
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    print(f"Summary for {output_dir}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <output_dir>")
        sys.exit(1)
    main(sys.argv[1])
