import wandb
try:
    api = wandb.Api()
    
    runs = [
        ("Low_Lambda", "khurramkhalil/STEER_PAPER_REPRO_FAST/runs/cm0clre6"),
        ("No_Steer", "khurramkhalil/STEER_PAPER_REPRO_FAST/runs/dfdulihx")
    ]
    
    print("Fetching WandB Results...")
    for name, path in runs:
        try:
            run = api.run(path)
            # Fetch summary
            summary = run.summary
            
            # Look for exact accuracy keys
            # The key might be "Sudoku-extreme.../exact_accuracy" or similar
            exact_acc = "N/A"
            for k, v in summary.items():
                if "exact_accuracy" in k and "train" not in k:
                    exact_acc = v
                    print(f"{name} ({k}): {v}")
            
            if exact_acc == "N/A":
                print(f"{name}: Could not find validation exact_accuracy in summary keys: {list(summary.keys())}")
                
        except Exception as e:
            print(f"Failed to fetch {name}: {e}")

except Exception as e:
    print(f"Global WandB Error: {e}")
