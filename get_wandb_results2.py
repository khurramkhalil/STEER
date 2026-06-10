import wandb
api = wandb.Api()
runs = api.runs("khurramkhalil/STEER_PAPER_EXACT")
for run in runs:
    try:
        val_acc = run.summary.get("all", {}).get("exact_accuracy", "N/A")
        train_acc = run.summary.get("train/exact_accuracy", "N/A")
        
        # Format as percentage if it's a float
        if isinstance(val_acc, float): val_acc = f"{val_acc*100:.2f}%"
        if isinstance(train_acc, float): train_acc = f"{train_acc*100:.2f}%"
        
        print(f"Run: {run.name}, Status: {run.state}, Train Exact Acc: {train_acc}, Val Exact Acc: {val_acc}")
    except Exception as e:
        print(f"Run: {run.name}, Error: {e}")
