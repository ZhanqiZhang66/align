import os
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

# CRITICAL: Set WandB mode BEFORE importing wandb
# For distributed training, torchrun sets RANK as an environment variable
rank = int(os.environ.get('RANK', 0))
if rank == 0:
    os.environ['WANDB_MODE'] = "online"  # Only rank 0 logs to WandB
else:
    os.environ['WANDB_MODE'] = "disabled"  # Disable WandB on all other ranks

# Set WandB API key (needed for all ranks, but only rank 0 will use it)
# You can also set this via environment variable WANDB_API_KEY
wandb_api_key = os.environ.get('WANDB_API_KEY')
if not wandb_api_key:
    # Set default API key if not in environment
    os.environ['WANDB_API_KEY'] = "57ee39f7bd565c158fe30f4e0466594e03bca347"

args = OmegaConf.load('rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()