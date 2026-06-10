import wandb
api = wandb.Api()
runs = api.runs("khurramkhalil/STEER_PAPER_EXACT")
for run in runs:
    exact_acc = "N/A"
    for k, v in run.summary.items():
        if "exact_accuracy" in k and "train" not in k:
            exact_acc = v
    print(f"Run: {run.name}, Exact Acc: {exact_acc}, State: {run.state}")
