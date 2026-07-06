import argparse
import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
import yaml


def _require(cfg: dict, path: str):
    cur = cfg
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU decoder from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '12355')}",
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(
                f"✅ DDP initialized: world_size={world_size}, "
                f"rank={rank}, local_rank={local_rank}"
            )

    if rank == 0:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    return rank, world_size, local_rank


def resolve_dataset_path(cfg: dict) -> str:
    paths_cfg = _require(cfg, "paths")
    if "dataset_path" in paths_cfg:
        return paths_cfg["dataset_path"]

    data_paths = _require(cfg, "paths.data_paths")
    data_path_key = _require(cfg, "paths.data_path_key")
    if data_path_key not in data_paths:
        raise KeyError(f"paths.data_path_key '{data_path_key}' not found in paths.data_paths")
    return data_paths[data_path_key]


def build_train_args(cfg: dict, seed: int, device: str, output_dir: str, dataset_path: str) -> dict:
    model_cfg = _require(cfg, "model")
    data_cfg = _require(cfg, "data")
    mask_cfg = _require(cfg, "masking")
    optim_cfg = _require(cfg, "optimization")

    training_cfg = _require(cfg, "training")
    other_cfg = cfg.get("other", {})

    model_name_base = _require(cfg, "training.model_name_base")
    model_name = f"{model_name_base}_seed_{seed}"

    args = {
        "seed": seed,
        "outputDir": output_dir,
        "datasetPath": dataset_path,
        "modelName": model_name,
        "device": device,
        "nInputFeatures": _require(cfg, "model.nInputFeatures"),
        "nClasses": _require(cfg, "model.nClasses"),
        "nUnits": _require(cfg, "model.nUnits"),
        "nLayers": _require(cfg, "model.nLayers"),
        "dropout": _require(cfg, "model.dropout"),
        "input_dropout": _require(cfg, "model.input_dropout"),
        "bidirectional": _require(cfg, "model.bidirectional"),
        "whiteNoiseSD": _require(cfg, "data.whiteNoiseSD"),
        "constantOffsetSD": _require(cfg, "data.constantOffsetSD"),
        "gaussianSmoothWidth": _require(cfg, "data.gaussianSmoothWidth"),
        "strideLen": _require(cfg, "data.strideLen"),
        "kernelLen": _require(cfg, "data.kernelLen"),
        "restricted_days": _require(cfg, "data.restricted_days"),
        "maxDay": _require(cfg, "data.maxDay"),
        "nDays": _require(cfg, "data.nDays"),
        "AdamW": _require(cfg, "optimization.AdamW"),
        "SOAP": optim_cfg.get("SOAP", False),
        "lrStart": _require(cfg, "optimization.lrStart"),
        "lrEnd": _require(cfg, "optimization.lrEnd"),
        "l2_decay": _require(cfg, "optimization.l2_decay"),
        "beta1": _require(cfg, "optimization.beta1"),
        "beta2": _require(cfg, "optimization.beta2"),
        "learning_scheduler": _require(cfg, "optimization.learning_scheduler"),
        "milestones": _require(cfg, "optimization.milestones"),
        "gamma": _require(cfg, "optimization.gamma"),
        "n_epochs": _require(cfg, "optimization.n_epochs"),
        "batchSize": _require(cfg, "optimization.batchSize"),
        "load_pretrained_model": training_cfg.get("checkpoint_dir", ""),
        "wandb_id": training_cfg.get("wandb_id", ""),
        "start_epoch": training_cfg.get("resume_from_epoch", 0),
        "ventral_6v_only": other_cfg.get("ventral_6v_only", False),
        "wandb_project": other_cfg.get("wandb_project", "Neural Decoder"),
        "wandb_entity": other_cfg.get("wandb_entity", None),
        "max_mask_pct": _require(cfg, "masking.max_mask_pct"),
        "num_masks": _require(cfg, "masking.num_masks"),
        "linderman_lab": _require(cfg, "masking.linderman_lab"),
        "consistency": _require(cfg, "masking.consistency"),
    }

    extra_args = cfg.get("trainer_args", {})
    if extra_args:
        args.update(extra_args)

    return args


def save_config_snapshot(config_path: str, output_dir: str, rank: int) -> None:
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "config_used.yaml")
    shutil.copy2(config_path, target_path)


def main() -> None:
    cli = parse_cli()
    cfg = load_config(cli.config)

    rank, _, local_rank = init_distributed()

    wandb_api_key = cfg.get("other", {}).get("wandb_api_key")
    if wandb_api_key and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)

    from neural_decoder.neural_decoder_trainer import trainModel
    from neural_decoder.model import GRUDecoder

    base_paths = _require(cfg, "paths.base_paths")
    server = _require(cfg, "paths.server")
    if server not in base_paths:
        raise KeyError(f"paths.server '{server}' not found in paths.base_paths")

    output_subdir = cfg.get("paths", {}).get("output_subdir", "outputs")
    dataset_path = resolve_dataset_path(cfg)

    seed_list = _require(cfg, "training.seed_list")
    model_name_base = _require(cfg, "training.model_name_base")

    local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if rank == 0:
        print(f"Using dataset: {dataset_path}")

    for seed in seed_list:
        model_name = f"{model_name_base}_seed_{seed}"
        output_dir = os.path.join(base_paths[server], output_subdir, model_name)

        args = build_train_args(
            cfg=cfg,
            seed=seed,
            device=device,
            output_dir=output_dir,
            dataset_path=dataset_path,
        )

        save_config_snapshot(cli.config, output_dir, rank)

        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])

        model = GRUDecoder(
            neural_dim=args["nInputFeatures"],
            n_classes=args["nClasses"],
            hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"],
            nDays=args["nDays"],
            dropout=args["dropout"],
            input_dropout=args["input_dropout"],
            device=args["device"],
            strideLen=args["strideLen"],
            kernelLen=args["kernelLen"],
            gaussianSmoothWidth=args["gaussianSmoothWidth"],
            bidirectional=args["bidirectional"],
            max_mask_pct=args["max_mask_pct"],
            num_masks=args["num_masks"],
            linderman_lab=args["linderman_lab"],
        ).to(args["device"])

        if rank == 0:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total:,}")
            print(f"Trainable parameters: {trainable:,}")

        trainModel(args, model)


if __name__ == "__main__":
    main()
