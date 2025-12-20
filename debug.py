import wandb

wandb.init()

for i in range(10):
    wandb.log({"step": i})
