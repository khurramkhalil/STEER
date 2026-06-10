import wandb
api = wandb.Api()
runs = api.runs("khurramkhalil/STEER_PAPER_EXACT")
for run in runs:
    print(f"Run: {run.name}")
    for k, v in run.summary.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if "exact" in sub_k or "accuracy" in sub_k:
                    print(f"  {k}/{sub_k}: {sub_v}")
        elif "exact" in k or "accuracy" in k:
            if "train" not in k:
                print(f"  {k}: {v}")
