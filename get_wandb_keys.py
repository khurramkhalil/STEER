import wandb
api = wandb.Api()
runs = api.runs("khurramkhalil/STEER_PAPER_EXACT")
for run in runs:
    print(f"Run: {run.name}")
    print(f"Summary keys: {list(run.summary.keys())[:10]}...") # Print first 10 keys
    for k, v in run.summary.items():
        if "accuracy" in k or "exact" in k:
            print(f"  {k}: {v}")
    break
