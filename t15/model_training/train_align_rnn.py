"""
Train the MDAN (multi-domain adversarial) model using rnn_align.yaml.
Uses manifest-based train/val/target split and domain adversarial loss when
source_days and target_days are set; otherwise trains CTC-only like the baseline.

Single GPU:
  cd /victoriapvc/repos/brain2text-t15/model_training && python train_model_mdan.py

With sweep config:
  python model_training/train_model_mdan.py --config configs/sweep_000_20250218.yaml

DDP (all 4 GPUs):
  cd /victoriapvc/repos/brain2text-t15/model_training && torchrun --nproc_per_node=4 train_model_mdan.py
"""
import argparse
import os


def _get_base_path():
    """Base path for data/outputs/tmp. Prefer BRAIN2TEXT_BASE; else repo root (or /victoriapvc if repo is under it)."""
    if os.environ.get('BRAIN2TEXT_BASE'):
        return os.environ['BRAIN2TEXT_BASE']
    path = os.path.dirname(os.path.abspath(__file__))
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, '.git')):
            if path.startswith('/victoriapvc'):
                return '/victoriapvc'
            return path
        path = os.path.dirname(path)
    return '/victoriapvc'


_BASE = _get_base_path()
if not os.environ.get('TMPDIR'):
    _tmp = os.path.join(_BASE, 'tmp')
    os.makedirs(_tmp, exist_ok=True)
    os.environ['TMPDIR'] = _tmp

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

if rank == 0:
    os.environ.setdefault('WANDB_MODE', 'online')
else:
    os.environ['WANDB_MODE'] = 'disabled'
if not os.environ.get('WANDB_API_KEY'):
    os.environ['WANDB_API_KEY'] = "c8d920049c36502840f7e96be79e3224286da5c7"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MDAN model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML (default: rnn_align.yaml in script dir)',
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='Comma-separated list of seeds to run; overrides seed and dataset.seed',
    )
    args_cli = parser.parse_args()
    if args_cli.config:
        config_path = os.path.abspath(args_cli.config)
    else:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rnn_align.yaml')
    args = OmegaConf.load(config_path)
    # Resolve /victoriapvc in config when base is not /victoriapvc (e.g. auto-detected repo root on server)
    _base = _get_base_path()
    if _base != '/victoriapvc':
        _container = OmegaConf.to_container(args, resolve=False)
        def _replace(obj):
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_replace(v) for v in obj]
            if isinstance(obj, str) and obj.startswith('/victoriapvc'):
                return _base + obj[len('/victoriapvc'):]
            return obj
        args = OmegaConf.create(_replace(_container))
    # Determine which seeds to run
    if args_cli.seeds:
        seeds = [int(s) for s in args_cli.seeds.split(',') if s.strip()]
    else:
        # Fall back to the single seed from the config
        seeds = [int(args.seed)] if 'seed' in args else [0]
    base_wandb_name = getattr(args, 'wandb_name', None)
    base_output_dir = args.get('output_dir', None)
    base_checkpoint_dir = args.get('checkpoint_dir', base_output_dir)
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        OmegaConf.update(args, 'rank', rank, merge=False)
        OmegaConf.update(args, 'local_rank', local_rank, merge=False)
        OmegaConf.update(args, 'world_size', world_size, merge=False)
    from rnn_trainer_align import BrainToTextDecoder_Trainer
    for seed in seeds:
        # Log which seed is starting (rank 0 only to avoid spam)
        if rank == 0:
            print(f'=== Starting training for seed {seed} ===')

        OmegaConf.update(args, 'seed', seed, merge=False)
        # Also sync dataset.seed if present
        try:
            if 'dataset' in args and 'seed' in args.dataset:
                OmegaConf.update(args, 'dataset.seed', seed, merge=False)
        except Exception:
            pass

        # Use a separate output/checkpoint directory per seed so files don't overwrite each other.
        if base_output_dir is not None:
            seed_output_dir = os.path.join(base_output_dir, f'seed_{seed}')
            OmegaConf.update(args, 'output_dir', seed_output_dir, merge=False)
        if base_checkpoint_dir is not None:
            seed_checkpoint_dir = os.path.join(base_checkpoint_dir, f'seed_{seed}')
            OmegaConf.update(args, 'checkpoint_dir', seed_checkpoint_dir, merge=False)

        # Give each seed a distinct WandB run name if possible
        if base_wandb_name:
            OmegaConf.update(args, 'wandb_name', f'{base_wandb_name}_seed{seed}', merge=False)

        trainer = BrainToTextDecoder_Trainer(args)
        metrics = trainer.train()

        if rank == 0:
            print(f'=== Finished training for seed {seed} ===')
    if world_size > 1:
        dist.destroy_process_group()
