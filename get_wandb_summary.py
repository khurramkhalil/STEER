import wandb
api = wandb.Api()
runs = api.runs("khurramkhalil/STEER_PAPER_EXACT")
for run in runs:
    if run.name == "L1.0_E0.1":
        print(f"Run: {run.name}")
        for k, v in run.summary.items():
            print(f"Key: {k}, Type: {type(v)}, Value: {v}")
        break
