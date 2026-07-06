"""Train the transformer MDAN phoneme decoder.

Single GPU:
  cd model_training && python train_model_transformer_mdan.py
  cd model_training && python train_model_transformer_mdan.py --config /abs/path/to/transformer_align.yaml

DDP (e.g. 4 GPUs):
  cd model_training && torchrun --nproc_per_node=4 train_model_transformer_mdan.py
"""
import os
import argparse


def _get_base_path():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.environ.get('B2TXT_TRANSFORMER_MDAN_CONFIG', 'model_training/transformer_align.yaml'),
        help='Path to transformer MDAN YAML config file.',
    )
    cli_args = parser.parse_args()

    args = OmegaConf.load(cli_args.config)

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

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        OmegaConf.update(args, 'rank', rank, merge=False)
        OmegaConf.update(args, 'local_rank', local_rank, merge=False)
        OmegaConf.update(args, 'world_size', world_size, merge=False)

    from transformer_trainer_align import BrainToTextTransformerMDAN_Trainer

    trainer = BrainToTextTransformerMDAN_Trainer(args)
    metrics = trainer.train()
    if world_size > 1:
        dist.destroy_process_group()
