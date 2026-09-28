import os
import subprocess
import json
import matplotlib.pyplot as plt

def main():
    seeds = [17, 42, 100]
    configs = ["configs/baseline_main.yaml", "configs/twotimescale_main.yaml"]
    names = ["baseline", "two_timescale"]
    
    results = {name: {"acc": [], "mem": [], "time": [], "params": []} for name in names}
    
    # Run experiments
    for name, config in zip(names, configs):
        for seed in seeds:
            out_dir = f"outputs/{name}_seed{seed}"
            print(f"Running {name} with seed {seed}...")
            
            # Using updated train CLI
            cmd = [
                "python", "-m", "recurrent_memory.cli", "train", 
                "--config", config, 
                "--seed", str(seed), 
                "--output", out_dir
            ]
            
            subprocess.run(cmd, check=True)
            
            # Read metrics
            with open(os.path.join(out_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)
                
            results[name]["acc"].append(metrics["test_acc"])
            results[name]["mem"].append(metrics["peak_memory_mb"])
            results[name]["time"].append(metrics["runtime_sec"])
            results[name]["params"].append(metrics.get("trainable_parameters", 0))

    # Aggregate and plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    metrics_to_plot = [
        ("Accuracy", "acc", "Test Accuracy"),
        ("Peak Memory (MB)", "mem", "Peak GPU Memory (MB)"),
        ("Runtime (s)", "time", "Total Runtime (s)"),
        ("Trainable Params", "params", "Trainable Params")
    ]
    
    for i, (title, key, ylabel) in enumerate(metrics_to_plot):
        data = [results[n][key] for n in names]
        axes[i].boxplot(data, labels=names)
        axes[i].set_title(title)
        axes[i].set_ylabel(ylabel)
        
        # Print means
        print(f"\n{title} Means:")
        for n in names:
            mean_val = sum(results[n][key]) / len(seeds)
            print(f"  {n}: {mean_val:.4f}")
            
    plt.tight_layout()
    plot_path = "outputs/experiment_results.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()
