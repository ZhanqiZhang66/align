"""Transformer MDAN trainer — ALIGN-style domain adaptation for BiT_Phoneme.

Implements the MDAN modules from ALIGN (arXiv:2603.18299):
  - Per-source-session binary discriminator heads (truely_mdan)
  - Gradient reversal layer on intermediate transformer representation
  - Sinusoidal alpha schedule (phase=16 for transformer)
  - Temporal Stretch Augmentation (TSA) on source and target data
  - Two-step backward to reduce GPU memory pressure

Falls back to standard CTC training when source_days/target_days are absent.
"""

import json
import logging
import math
import os
import pathlib
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F_torch
import torchaudio.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_augmentations import gauss_smooth
from dataset import (
    BrainToTextDataset,
    BrainToTextTrialDataset,
    ProlongedDataset,
    StretchSqueezeDataset,
    collate_trial_batches,
    generate_prolonged_samples,
    train_test_split_indicies,
)
from dann import (
    DomainDiscriminator,
    grad_reverse,
    masked_mean_pool,
    randomly_mask_channelsteps,
)
from transformer_model import BiT_Phoneme

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch._dynamo.config.cache_size_limit = 64


# ----------------------------------------------------------------- schedules
def dann_alpha(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None) -> float:
    """Standard DANN sigmoid schedule: α ∈ [0, 1]."""
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def dann_alpha_alt(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None, phase: int = 16) -> float:
    """ALIGN sinusoidal schedule: α = 0.5 + 0.5·sin(phase·π·p − π/2)."""
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    return float(0.5 + 0.5 * math.sin(phase * math.pi * p - math.pi / 2))


# --------------------------------------------------------------- trainer
class BrainToTextTransformerMDAN_Trainer:
    """Train BiT_Phoneme with ALIGN-style MDAN domain adaptation."""

    def _seed_everything(self, seed_value, label: str):
        if seed_value is None or int(seed_value) == -1:
            return None
        base_seed = int(seed_value)
        seed = base_seed + self.rank
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if self.rank == 0 and self.logger is not None:
            self.logger.info(f"{label} seed: base={base_seed}, rank_adjusted={seed}")
        return seed

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None

        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        # MDAN fields
        self.domain_disc = None
        self.disc_optimizer = None
        self.train_loader_source = None
        self.train_loader_target = None
        self.day_index_to_domain_id = None
        self.n_domains = 0
        self.truely_mdan = False

        self.distributed = bool(self.args.get("world_size", 1) > 1)
        self.rank = int(self.args.get("rank", 0))
        self.world_size = int(self.args.get("world_size", 1))
        self.local_rank = int(self.args.get("local_rank", 0))
        base_seed = self.args.get("seed", -1)
        self.init_seed = self.args.get("init_seed", base_seed)
        self.train_seed = self.args.get("train_seed", base_seed)

        # Detect MDAN mode (can be overridden by config flag `use_mdan`)
        _src = self.args.get("source_days", None)
        _tgt = self.args.get("target_days", None)
        autodetect_mdan = (
            _src is not None and len(_src) > 0
            and _tgt is not None and len(_tgt) > 0
        )
        use_mdan_cfg = self.args.get("use_mdan", None)
        self.use_mdan = autodetect_mdan if use_mdan_cfg is None else bool(use_mdan_cfg)

        self.transform_args = self.args["dataset"]["data_transforms"]
        self.per_device_batch_size = self.args["dataset"]["batch_size"] // max(self.world_size, 1)

        if args["mode"] == "train":
            os.makedirs(self.args["output_dir"], exist_ok=True)

        ckpt = self.args.get("checkpoint_dir")
        if ckpt is None or (isinstance(ckpt, str) and ckpt.strip() == ""):
            self.args["checkpoint_dir"] = self.args["output_dir"]

        if args["save_best_checkpoint"] or args["save_all_val_steps"] or args["save_final_model"]:
            os.makedirs(self.args["checkpoint_dir"], exist_ok=True)

        # ------------------------------------------------------------ logger
        self.logger = logging.getLogger(__name__)
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter(fmt="%(asctime)s: %(message)s")

        if args["mode"] == "train":
            fh = logging.FileHandler(str(pathlib.Path(self.args["output_dir"], "training_log")))
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        # ------------------------------------------------------------ wandb
        self.wandb_initialized = False
        if args["mode"] == "train":
            wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
            if wandb_mode != "disabled":
                wandb_project = self.args.get("wandb_project", "Brain-to-Text Transformer MDAN")
                wandb_entity = self.args.get("wandb_entity", None)
                wandb_name = self.args.get(
                    "wandb_name",
                    self.args.get("output_dir", "transformer_mdan").split("/")[-1],
                )
                if os.environ.get("WANDB_API_KEY"):
                    try:
                        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
                    except Exception as e:
                        self.logger.warning(f"WandB login warning: {e}")
                else:
                    self.logger.warning("WANDB_API_KEY not found in environment")

                init_kwargs = {
                    "project": wandb_project,
                    "config": OmegaConf.to_container(args, resolve=True),
                    "name": wandb_name,
                    "mode": "online",
                }
                if wandb_entity is not None:
                    init_kwargs["entity"] = wandb_entity
                if self.args.get("wandb_id") and len(self.args.get("wandb_id", "")) > 0:
                    init_kwargs["resume"] = "must"
                    init_kwargs["id"] = self.args["wandb_id"]

                try:
                    wandb.init(**init_kwargs)
                    if wandb.run is not None and wandb.run.url is not None:
                        self.wandb_initialized = True
                        self.logger.info(f"WandB ONLINE: {wandb.run.url}")
                    else:
                        raise Exception("WandB run created but no URL")
                except wandb.errors.CommError as e:
                    self.logger.warning(f"WandB permission error: {e}")
                    time.sleep(2)
                    if wandb.run is not None and wandb.run.url is not None:
                        self.wandb_initialized = True
                    else:
                        raise RuntimeError("WandB failed to initialize ONLINE.")
                except Exception as e:
                    self.logger.error(f"WandB error: {e}")
                    time.sleep(1)
                    if wandb.run is None or wandb.run.url is None:
                        raise RuntimeError("WandB failed to initialize ONLINE.")
                    self.wandb_initialized = True
            else:
                self.logger.info("WandB disabled via WANDB_MODE.")

        # ------------------------------------------------------------ device
        if torch.cuda.is_available():
            gpu_num = self.local_rank if self.distributed else self.args.get("gpu_number", 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                gpu_num = 0
            if gpu_num > torch.cuda.device_count() - 1:
                gpu_num = 0
            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                _ = torch.tensor([1.0]).to(self.device) * 2
            except Exception as e:
                self.logger.error(f"CUDA init error: {e}; falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.rank == 0:
            self.logger.info(
                f"Using device: {self.device}"
                + (f" (DDP rank {self.rank}/{self.world_size})" if self.distributed else "")
            )

        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = torch.bfloat16
        if self.args.get("use_amp", False):
            if self.autocast_device_type == "cuda" and not torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.float16
            if self.rank == 0:
                self.logger.info(f"AMP autocast dtype: {self.amp_dtype}")

        # ------------------------------------------------------------ seed
        self._seed_everything(self.init_seed, "Initialization")

        # ------------------------------------------------------------ model
        m = self.args["model"]
        n_classes_no_blank = self.args["dataset"]["n_classes"] - 1
        self.model = BiT_Phoneme(
            patch_size=m["patch_size"],
            patch_stride=m.get("patch_stride"),
            neural_dim=m["n_input_features"],
            dim=m["dim"],
            depth=m["depth"],
            heads=m["heads"],
            dim_head=m["dim_head"],
            mlp_dim_ratio=m["mlp_dim_ratio"],
            dropout=m["dropout"],
            attn_dropout=m["attn_dropout"],
            input_dropout=m["input_dropout"],
            gaussian_smooth_width=m["gaussian_smooth_width"],
            gaussian_smooth_size=m.get("gaussian_smooth_size", 21),
            n_classes=n_classes_no_blank,
            T5_style_pos=m["T5_style_pos"],
            max_mask_pct=m["max_mask_pct"],
            num_masks=m["num_masks"],
            mask_token_zeros=m["mask_token_zeros"],
            max_rel_dist=m.get("max_rel_dist", 200),
            use_gradient_checkpointing=m.get("use_gradient_checkpointing", False),
        )

        if self.args["dataset"].get("use_torch_compile", False):
            self.model = torch.compile(self.model)

        self.logger.info("Initialized BiT_Phoneme decoding model")
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {n_params:,} parameters")

        # ------------------------------------------------------------ data
        manifest_path = self.args.get("manifest_path")
        use_manifest = (
            manifest_path is not None
            and str(manifest_path).strip() not in ("", "None")
            and os.path.exists(str(manifest_path))
        )

        train_trials = {}
        val_trials = {}
        competition_trials = {}

        if use_manifest:
            self.logger.info(f"Loading datasets from manifest: {manifest_path}")
            with open(manifest_path, "rb") as f:
                manifest = pickle.load(f)
            train_trials_raw = manifest.get("train_trial_indicies", {})
            val_trials_raw = manifest.get("val_trial_indicies", {})
            comp_trials_raw = manifest.get("competition_trial_indicies", {})
            training_days = manifest.get("training_days", [])
            manifest_target_days = manifest.get("target_days", [])
            sessions = self.args["dataset"]["sessions"]

            for rel_idx, td in train_trials_raw.items():
                if rel_idx < len(training_days):
                    sn = training_days[rel_idx]
                    abs_idx = sessions.index(sn) if sn in sessions else rel_idx
                    train_trials[abs_idx] = td
            for rel_idx, td in val_trials_raw.items():
                if rel_idx < len(manifest_target_days):
                    sn = manifest_target_days[rel_idx]
                    abs_idx = sessions.index(sn) if sn in sessions else rel_idx + len(training_days)
                    val_trials[abs_idx] = td
            for rel_idx, td in comp_trials_raw.items():
                if rel_idx < len(manifest_target_days):
                    sn = manifest_target_days[rel_idx]
                    abs_idx = sessions.index(sn) if sn in sessions else rel_idx + len(training_days)
                    competition_trials[abs_idx] = td
            self.logger.info(f"Loaded {len(train_trials)} train, {len(val_trials)} val sessions")
        else:
            self.logger.info("Loading from dataset_dir HDF5...")
            train_paths = [
                os.path.join(self.args["dataset"]["dataset_dir"], s, "data_train.hdf5")
                for s in self.args["dataset"]["sessions"]
            ]
            val_paths = [
                os.path.join(self.args["dataset"]["dataset_dir"], s, "data_val.hdf5")
                for s in self.args["dataset"]["sessions"]
            ]
            train_trials, _ = train_test_split_indicies(
                file_paths=train_paths, test_percentage=0,
                seed=self.args["dataset"]["seed"], bad_trials_dict=None,
            )
            _, val_trials = train_test_split_indicies(
                file_paths=val_paths, test_percentage=1,
                seed=self.args["dataset"]["seed"], bad_trials_dict=None,
            )

        with open(os.path.join(self.args["output_dir"], "train_val_trials.json"), "w") as f:
            json.dump({"train": train_trials, "val": val_trials, "competition": competition_trials}, f)

        feature_subset = self.args["dataset"].get("feature_subset")

        # ---- validation loader (same for MDAN and baseline) ----
        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split="test", days_per_batch=None, n_batches=None,
            batch_size=self.args["dataset"]["batch_size"],
            must_include_days=None,
            random_seed=self.args["dataset"]["seed"],
            feature_subset=feature_subset,
        )
        vw = self.args["dataset"].get("num_val_dataloader_workers", 0)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=None, shuffle=False,
            num_workers=vw, pin_memory=True,
            prefetch_factor=2 if vw > 0 else None,
            persistent_workers=vw > 0,
        )

        # ---- source / target loaders (MDAN) or single loader (baseline) ----
        if self.use_mdan:
            source_days = list(self.args["source_days"])
            target_days = list(self.args["target_days"])

            if use_manifest and len(competition_trials) > 0:
                train_trials_source = {d: train_trials[d] for d in source_days if d in train_trials and len(train_trials[d]["trials"]) > 0}
                train_trials_target = {d: competition_trials[d] for d in target_days if d in competition_trials and len(competition_trials[d]["trials"]) > 0}
                self.logger.info(f"MDAN: source from train ({len(train_trials_source)} sessions), target from competition ({len(train_trials_target)} sessions)")
            else:
                train_trials_source = {d: train_trials[d] for d in source_days if d in train_trials and len(train_trials[d]["trials"]) > 0}
                train_trials_target = {d: train_trials[d] for d in target_days if d in train_trials and len(train_trials[d]["trials"]) > 0}
                self.logger.info(f"MDAN: source from train ({len(train_trials_source)} sessions), target from train ({len(train_trials_target)} sessions)")

            if len(train_trials_source) == 0 or len(train_trials_target) == 0:
                raise ValueError("MDAN: source_days or target_days yielded no training trials. Check source_days/target_days match session indices.")

            self.day_index_to_domain_id = {d: idx for idx, d in enumerate(source_days)}
            # Pre-built GPU lookup table so we can map day indices → domain ids
            # without a per-batch GPU→CPU→GPU round-trip + Python loop.
            max_day_idx = max(self.day_index_to_domain_id.keys())
            lut = torch.full((max_day_idx + 1,), -1, dtype=torch.long)
            for k, v in self.day_index_to_domain_id.items():
                lut[k] = v
            self._domain_id_lut = lut.to(self.device)

            include_stretched = self.args.get("include_stretched_samples", False)
            include_prolonged = self.args.get("include_prolonged_samples", False)
            include_original = self.args.get("include_original", True)
            stretch_range = self.args.get("stretch_range", 2.0)
            prolonged_combined_range = tuple(self.args.get("prolonged_combined_range", [1, 10]))
            prolonged_sample_size = self.args.get("prolonged_sample_size")
            use_trial_level_source = include_stretched or include_prolonged

            if use_trial_level_source:
                base_src_ds = BrainToTextTrialDataset(train_trials_source, feature_subset=feature_subset)
                src_parts = []
                if include_stretched:
                    src_parts.append(StretchSqueezeDataset(base_src_ds, stretch_range=stretch_range))
                if include_prolonged:
                    prolonged_list = generate_prolonged_samples(
                        base_src_ds,
                        combined_range=prolonged_combined_range,
                        stretch_range=stretch_range,
                        sample_size=prolonged_sample_size,
                    )
                    src_parts.append(ProlongedDataset(prolonged_list))
                if include_original:
                    src_parts.append(base_src_ds)
                self.train_dataset_source = ConcatDataset(src_parts)
                self.logger.info(f"MDAN source: stretch={include_stretched} prolonged={include_prolonged} original={include_original} len={len(self.train_dataset_source)}")
            else:
                self.train_dataset_source = BrainToTextDataset(
                    trial_indicies=train_trials_source,
                    split="train",
                    days_per_batch=min(self.args["dataset"]["days_per_batch"], len(source_days)),
                    n_batches=self.args["num_training_batches"],
                    batch_size=self.per_device_batch_size,
                    must_include_days=None,
                    random_seed=self.args["dataset"]["seed"],
                    feature_subset=feature_subset,
                )

            src_sampler = None
            src_shuffle = self.args["dataset"]["loader_shuffle"]
            if self.distributed:
                src_sampler = DistributedSampler(self.train_dataset_source, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                src_shuffle = False
            nw = self.args["dataset"]["num_dataloader_workers"]
            pf = self.args["dataset"].get("prefetch_factor", 2)
            persistent = nw > 0 and self.args["dataset"].get("persistent_workers", True)
            self.train_loader_source = DataLoader(
                self.train_dataset_source,
                batch_size=self.per_device_batch_size if use_trial_level_source else None,
                shuffle=src_shuffle, sampler=src_sampler,
                num_workers=nw, pin_memory=True,
                prefetch_factor=pf if nw > 0 else None,
                persistent_workers=persistent,
                collate_fn=collate_trial_batches if use_trial_level_source else None,
            )

            # Target loader
            if include_stretched:
                base_tgt_ds = BrainToTextTrialDataset(train_trials_target, feature_subset=feature_subset)
                stretched_tgt = StretchSqueezeDataset(base_tgt_ds, stretch_range=stretch_range)
                self.train_dataset_target = ConcatDataset([base_tgt_ds, stretched_tgt]) if include_original else stretched_tgt
                self.logger.info(f"MDAN target: stretch=True original={include_original} len={len(self.train_dataset_target)}")
            else:
                n_batches_target = max(1, self.args["num_training_batches"] // 2)
                self.train_dataset_target = BrainToTextDataset(
                    trial_indicies=train_trials_target,
                    split="train",
                    days_per_batch=min(self.args["dataset"]["days_per_batch"], len(target_days)),
                    n_batches=n_batches_target,
                    batch_size=self.per_device_batch_size,
                    must_include_days=None,
                    random_seed=self.args["dataset"]["seed"],
                    feature_subset=feature_subset,
                )

            tgt_sampler = None
            tgt_shuffle = True
            if self.distributed:
                tgt_sampler = DistributedSampler(self.train_dataset_target, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                tgt_shuffle = False
            self.train_loader_target = DataLoader(
                self.train_dataset_target,
                batch_size=self.per_device_batch_size if include_stretched else None,
                shuffle=tgt_shuffle, sampler=tgt_sampler,
                num_workers=nw, pin_memory=True,
                prefetch_factor=pf if nw > 0 else None,
                persistent_workers=persistent,
                collate_fn=collate_trial_batches if include_stretched else None,
            )
        else:
            # Baseline mode: single train loader
            self.train_dataset = BrainToTextDataset(
                trial_indicies=train_trials,
                split="train",
                days_per_batch=self.args["dataset"]["days_per_batch"],
                n_batches=self.args["num_training_batches"],
                batch_size=self.args["dataset"]["batch_size"],
                must_include_days=None,
                random_seed=self.args["dataset"]["seed"],
                feature_subset=feature_subset,
            )
            nw = self.args["dataset"]["num_dataloader_workers"]
            pf = self.args["dataset"].get("prefetch_factor", 2)
            persistent = nw > 0 and self.args["dataset"].get("persistent_workers", True)
            train_sampler = None
            train_shuffle = self.args["dataset"]["loader_shuffle"]
            if self.distributed:
                train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                train_shuffle = False
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=None, shuffle=train_shuffle, sampler=train_sampler,
                num_workers=nw, pin_memory=True,
                prefetch_factor=pf if nw > 0 else None,
                persistent_workers=persistent,
            )

        # ---- MDAN discriminators ----
        if self.use_mdan:
            self._seed_everything(self.init_seed, "Discriminator initialization")
            source_days = list(self.args["source_days"])
            self.n_domains = len(source_days)
            self.truely_mdan = bool(self.args.get("truely_mdan", True))
            rep_dim = self.model.dim  # transformer embedding dim (e.g. 384)
            domain_hidden = int(self.args.get("dann_hidden", 256))
            domain_dropout = float(self.args.get("domain_dropout", 0.1))
            linear_discriminator = bool(self.args.get("linear_discriminator", False))
            dann_lr_multiplier = float(self.args.get("dann_lr_multiplier", 0.6))
            dann_weight_decay = self.args.get("dann_weight_decay", self.args["weight_decay"])

            if self.truely_mdan:
                self.domain_disc = torch.nn.ModuleList([
                    DomainDiscriminator(
                        in_dim=rep_dim,
                        hidden_dim=domain_hidden,
                        n_domains=1,
                        dropout=domain_dropout,
                        linear_discriminator=linear_discriminator,
                    ) for _ in range(self.n_domains)
                ]).to(self.device)
            else:
                self.domain_disc = DomainDiscriminator(
                    in_dim=rep_dim,
                    hidden_dim=domain_hidden,
                    n_domains=self.n_domains + 1,
                    dropout=domain_dropout,
                    linear_discriminator=linear_discriminator,
                ).to(self.device)

            disc_params = list(self.domain_disc.parameters())
            disc_lr = self.args["lr_max"] * dann_lr_multiplier
            disc_wd = dann_weight_decay if dann_weight_decay is not None else self.args["weight_decay"]
            use_fused = self.device.type in ("cuda", "xpu", "privateuseone")
            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=disc_lr,
                weight_decay=disc_wd,
                betas=(self.args["beta0"], self.args["beta1"]),
                fused=use_fused,
            )
            self.logger.info(
                f"MDAN enabled: source={list(self.args['source_days'])}, target={list(self.args['target_days'])}, "
                f"n_domains={self.n_domains}, truely_mdan={self.truely_mdan}, rep_dim={rep_dim}"
            )

        if self.args["init_from_checkpoint"]:
            self.load_model_checkpoint(self.args["init_checkpoint_path"])

        self.model.to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=self.use_mdan)

        self.optimizer = self.create_optimizer()

        if self.args["lr_scheduler_type"] == "linear":
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args["lr_min"] / self.args["lr_max"],
                total_iters=self.args["lr_decay_steps"],
            )
        elif self.args["lr_scheduler_type"] == 'multistep': 
            print("Multistep scheduler")
            self.learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.args['milestones'],
                gamma=float(self.args['gamma']),
            )
        elif self.args["lr_scheduler_type"] == "cosine":
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid lr scheduler: {self.args['lr_scheduler_type']}")

        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
        self._seed_everything(self.train_seed, "Training")

        if self.wandb_initialized and wandb.run is not None:
            wandb.watch(self.model, log=None)

    # ----------------------------------------------------------------- optim
    def create_optimizer(self):
        def is_no_decay(name: str, param: torch.Tensor) -> bool:
            lname = name.lower()
            if name.endswith(".bias"):
                return True
            if "norm" in lname and param.dim() == 1:
                return True
            if "mask_token" in lname:
                return True
            if "rel_pos_bias" in lname:
                return True
            return False

        no_decay, decay = [], []
        for name, p in self.model.named_parameters():
            if is_no_decay(name, p):
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": no_decay, "weight_decay": 0.0, "group_type": "no_decay"},
            {"params": decay, "group_type": "decay"},
        ]
        use_fused = self.device.type in ("cuda", "xpu", "privateuseone")
        return torch.optim.AdamW(
            param_groups,
            lr=self.args["lr_max"],
            betas=(self.args["beta0"], self.args["beta1"]),
            eps=self.args["epsilon"],
            weight_decay=self.args["weight_decay"],
            fused=use_fused,
        )

    def create_cosine_lr_scheduler(self, optim):
        lr_max, lr_min = self.args["lr_max"], self.args["lr_min"]
        decay_steps = self.args["lr_decay_steps"]
        warmup = self.args["lr_warmup_steps"]
        min_ratio = lr_min / lr_max

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            if step < decay_steps:
                progress = float(step - warmup) / float(max(1, decay_steps - warmup))
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_ratio, min_ratio + (1 - min_ratio) * cosine)
            return min_ratio

        return LambdaLR(optim, [lr_lambda] * len(optim.param_groups), -1)

    # ----------------------------------------------------------------- ckpt
    def load_model_checkpoint(self, load_path):
        ckpt = torch.load(load_path, weights_only=False)
        m = self.model.module if self.distributed else self.model
        m.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.learning_rate_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_PER = ckpt["val_PER"]
        self.best_val_loss = ckpt.get("val_loss", torch.inf)
        if self.domain_disc is not None and "domain_disc_state_dict" in ckpt:
            self.domain_disc.load_state_dict(ckpt["domain_disc_state_dict"])
        if self.disc_optimizer is not None and "disc_optimizer_state_dict" in ckpt:
            self.disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
        self.model.to(self.device)
        for st in self.optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(self.device)
        if self.disc_optimizer is not None:
            for st in self.disc_optimizer.state.values():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(self.device)
        if self.rank == 0:
            self.logger.info("Loaded checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        m = self.model.module if self.distributed else self.model
        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            "val_PER": PER, "val_loss": loss,
        }
        if self.domain_disc is not None:
            ckpt["domain_disc_state_dict"] = self.domain_disc.state_dict()
        if self.disc_optimizer is not None:
            ckpt["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        torch.save(ckpt, save_path)
        if self.rank == 0:
            self.logger.info("Saved checkpoint: " + save_path)
            with open(os.path.join(self.args["checkpoint_dir"], "args.yaml"), "w") as f:
                OmegaConf.save(config=self.args, f=f)

    # -------------------------------------------------------------- aug
    def _adjusted_lens(self, n_time_steps: torch.Tensor) -> torch.Tensor:
        m = self.model.module if self.distributed else self.model
        return m.compute_length(n_time_steps)

    def transform_data(self, features, n_time_steps, mode="train"):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == "train":
            if self.transform_args["static_gain_std"] > 0:
                warp = torch.tile(torch.unsqueeze(torch.eye(channels), dim=0), (batch_size, 1, 1))
                warp += torch.randn_like(warp, device=self.device) * self.transform_args["static_gain_std"]
                features = torch.matmul(features, warp)
            if self.transform_args["white_noise_std"] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args["white_noise_std"]
            if self.transform_args["constant_offset_std"] > 0:
                features += (
                    torch.randn((batch_size, 1, channels), device=self.device)
                    * self.transform_args["constant_offset_std"]
                )
            if self.transform_args["random_walk_std"] > 0:
                features += torch.cumsum(
                    torch.randn(data_shape, device=self.device) * self.transform_args["random_walk_std"],
                    dim=self.transform_args["random_walk_axis"],
                )
            if self.transform_args["random_cut"] > 0:
                cut = np.random.randint(0, self.transform_args["random_cut"])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        if self.transform_args.get("smooth_data", False):
            features = gauss_smooth(
                inputs=features, device=self.device,
                smooth_kernel_std=self.transform_args["smooth_kernel_std"],
                smooth_kernel_size=self.transform_args["smooth_kernel_size"],
            )
        return features, n_time_steps

    # -------------------------------------------------------------- train
    def train(self):
        self.model.train()
        train_losses, val_losses, val_PERs, val_results = [], [], [], []
        val_steps_since_improvement = 0
        save_best = self.args.get("save_best_checkpoint", True)
        early_stopping = self.args.get("early_stopping", True)
        early_stopping_val_steps = self.args["early_stopping_val_steps"]

        train_start = time.time()
        num_training_batches = self.args["num_training_batches"]

        # MDAN hyper-params used throughout training loop
        rep_layer_idx = self.args.get("rep_layer_idx", None)
        lambda_domain = float(self.args.get("dann_lambda", 0.6))
        dann_warmup_steps = int(self.args.get("dann_warmup_steps", 5))
        _alpha_max = self.args.get("dann_alpha_max_steps")
        alpha_max_steps_cfg = int(_alpha_max) if _alpha_max is not None else None
        mean_pool_for_disc = bool(self.args.get("mean_pool_for_discriminator", False))
        lambda_tgt = float(self.args.get("lambda_tgt", 1.0))
        dom_loss_type = self.args.get("dom_loss_type", "mean")
        phase = int(self.args.get("phase", 16))

        train_loop_loader = self.train_loader_source if self.use_mdan else self.train_loader
        steps_per_epoch = len(train_loop_loader)
        num_epochs = max(1, (num_training_batches + steps_per_epoch - 1) // steps_per_epoch) if steps_per_epoch > 0 else 1

        total_steps = 0
        last_i = 0
        early_stop_flag = False

        for epoch in range(num_epochs):
            if self.distributed and hasattr(train_loop_loader, "sampler") and hasattr(train_loop_loader.sampler, "set_epoch"):
                train_loop_loader.sampler.set_epoch(epoch)
            if self.use_mdan and self.distributed and hasattr(self.train_loader_target, "sampler") and hasattr(self.train_loader_target.sampler, "set_epoch"):
                self.train_loader_target.sampler.set_epoch(epoch)

            target_iter = iter(self.train_loader_target) if self.use_mdan else None

            for i, batch in enumerate(train_loop_loader):
                self.model.train()
                if self.domain_disc is not None:
                    self.domain_disc.train()
                self.optimizer.zero_grad(set_to_none=True)
                if self.disc_optimizer is not None:
                    self.disc_optimizer.zero_grad(set_to_none=True)

                start = time.time()

                # ---------- source batch ----------
                features = batch["input_features"].to(self.device)
                labels = batch["seq_class_ids"].to(self.device)
                n_time_steps = batch["n_time_steps"].to(self.device)
                phone_seq_lens = batch["phone_seq_lens"].to(self.device)
                day_indicies = batch["day_indicies"].to(self.device)

                with torch.autocast(device_type=self.autocast_device_type, enabled=self.args["use_amp"], dtype=self.amp_dtype):
                    features, n_time_steps = self.transform_data(features, n_time_steps, "train")
                    adjusted_lens = self._adjusted_lens(n_time_steps)

                    if self.use_mdan:
                        logits, src_rep = self.model(features, n_time_steps, day_indicies, return_rep=True, rep_layer_idx=rep_layer_idx)
                    else:
                        logits = self.model(features, n_time_steps, day_indicies)

                # CTC must run in float32 for numerical stability (outside autocast).
                task_loss = self.ctc_loss(
                    log_probs=torch.permute(logits.float().log_softmax(2), [1, 0, 2]),
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens,
                )
                task_loss = torch.mean(task_loss)

                # ---------- MDAN domain loss ----------
                dom_loss = torch.tensor(0.0, device=self.device)
                current_lambda_domain = 0.0

                if self.use_mdan:
                    use_two_step = bool(self.args.get("mdan_two_step_backward", True))
                    alpha_max_steps = alpha_max_steps_cfg if alpha_max_steps_cfg is not None else num_training_batches
                    alpha_step = max(total_steps - dann_warmup_steps, 0)
                    is_warmup = total_steps < dann_warmup_steps

                    if is_warmup:
                        alpha = 0.0
                    elif self.args.get("dann_alpha_type", "alternative") == "alternative":
                        alpha = dann_alpha_alt(alpha_step, alpha_max_steps, phase=phase)
                    else:
                        alpha = dann_alpha(alpha_step, alpha_max_steps)

                    current_lambda_domain = 0.0 if is_warmup else lambda_domain

                    if use_two_step:
                        # Step 1: backward task loss, then detach source rep
                        task_loss.backward()
                        adjusted_lens_src = adjusted_lens
                        drop_prob = self.args.get("dann_dropout_channel_prob", 0.0)
                        if drop_prob > 0:
                            src_rep_det = randomly_mask_channelsteps(src_rep.detach(), adjusted_lens_src, drop_prob)
                        else:
                            src_rep_det = src_rep.detach()
                        if mean_pool_for_disc:
                            src_feat_det = masked_mean_pool(src_rep_det, adjusted_lens_src)
                        else:
                            src_feat_det = src_rep_det
                        src_domain_ids = self._domain_id_lut[day_indicies]
                        del features, logits, src_rep

                        src_feat_grl = src_feat_det if is_warmup else grad_reverse(src_feat_det, alpha=alpha)
                    else:
                        drop_prob = self.args.get("dann_dropout_channel_prob", 0.0)
                        if drop_prob > 0:
                            src_rep = randomly_mask_channelsteps(src_rep, adjusted_lens, drop_prob)
                        if mean_pool_for_disc:
                            src_feat = masked_mean_pool(src_rep, adjusted_lens)
                        else:
                            src_feat = src_rep
                        src_feat_grl = src_feat.detach() if is_warmup else grad_reverse(src_feat, alpha=alpha)
                        src_domain_ids = self._domain_id_lut[day_indicies]

                    # Fetch target batch
                    try:
                        tgt_batch = next(target_iter)
                    except (StopIteration, TypeError):
                        target_iter = iter(self.train_loader_target)
                        tgt_batch = next(target_iter)
                    tgt_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in tgt_batch.items()}
                    tgt_features = tgt_batch["input_features"]
                    tgt_n_time_steps = tgt_batch["n_time_steps"]

                    with torch.autocast(device_type=self.autocast_device_type, enabled=self.args["use_amp"], dtype=self.amp_dtype):
                        tgt_features, tgt_n_time_steps = self.transform_data(tgt_features, tgt_n_time_steps, "train")
                        adjusted_lens_tgt = self._adjusted_lens(tgt_n_time_steps)
                        _, tgt_rep = self.model(tgt_features, tgt_n_time_steps, tgt_batch["day_indicies"], return_rep=True, rep_layer_idx=rep_layer_idx)

                    drop_prob = self.args.get("dann_dropout_channel_prob", 0.0)
                    if drop_prob > 0:
                        tgt_rep = randomly_mask_channelsteps(tgt_rep, adjusted_lens_tgt, drop_prob)
                    if mean_pool_for_disc:
                        tgt_feat = masked_mean_pool(tgt_rep, adjusted_lens_tgt)
                    else:
                        tgt_feat = tgt_rep
                    tgt_feat_grl = tgt_feat.detach() if is_warmup else grad_reverse(tgt_feat, alpha=alpha)

                    # Compute domain loss
                    def hinge_loss(logits, is_source):
                        t = 1.0 if is_source else -1.0
                        return torch.clamp(1.0 - t * logits, min=0.0)

                    if current_lambda_domain > 0:
                        if self.truely_mdan:
                            # Hoist per-loop allocations (arange, valid-step masks)
                            # out of the K-discriminator loop.
                            if not mean_pool_for_disc:
                                Bs, Ts, _ = src_feat_grl.shape
                                adj_lens_s = adjusted_lens_src if use_two_step else adjusted_lens
                                src_valid = (
                                    torch.arange(Ts, device=self.device)[None, :]
                                    < adj_lens_s[:, None]
                                )  # (Bs, Ts)
                                Bt, Tt, _ = tgt_feat_grl.shape
                                tgt_valid = (
                                    torch.arange(Tt, device=self.device)[None, :]
                                    < adjusted_lens_tgt[:, None]
                                )  # (Bt, Tt)
                            else:
                                Bt = tgt_feat_grl.shape[0]

                            dom_losses = []
                            for disc_idx, disc in enumerate(self.domain_disc):
                                src_mask = (src_domain_ids == disc_idx)
                                if src_mask.any():
                                    if mean_pool_for_disc:
                                        logits_s = disc(src_feat_grl[src_mask].float()).view(-1)
                                    else:
                                        sel_s = src_mask[:, None] & src_valid
                                        logits_s = disc(src_feat_grl[sel_s].float()).view(-1)
                                    src_loss_i = hinge_loss(logits_s, True).mean()
                                else:
                                    src_loss_i = torch.tensor(0.0, device=self.device)

                                m_src = int(src_mask.sum().item())
                                if m_src > 0:
                                    if mean_pool_for_disc:
                                        Bt = tgt_feat_grl.shape[0]
                                    mm = min(m_src, Bt)
                                    tgt_idx = torch.randperm(Bt, device=self.device)[:mm]
                                    if mean_pool_for_disc:
                                        logits_t = disc(tgt_feat_grl[tgt_idx].float()).view(-1)
                                    else:
                                        sel_t = tgt_valid[tgt_idx]
                                        logits_t = disc(tgt_feat_grl[tgt_idx][sel_t].float()).view(-1)
                                    tgt_loss_i = hinge_loss(logits_t, False).mean()
                                else:
                                    tgt_loss_i = torch.tensor(0.0, device=self.device)

                                dom_losses.append(0.5 * (src_loss_i + lambda_tgt * tgt_loss_i))

                            if dom_losses:
                                dom_stack = torch.stack(dom_losses)
                                dom_loss = dom_stack.max() if dom_loss_type == "max" else dom_stack.mean()
                            else:
                                dom_loss = torch.tensor(0.0, device=self.device)
                        else:
                            # Single multi-class discriminator (standard DANN)
                            src_logits = self.domain_disc(src_feat_grl.float())
                            if mean_pool_for_disc:
                                src_dom_loss = F_torch.cross_entropy(src_logits, src_domain_ids, reduction="mean")
                            else:
                                Bs, Ts, _ = src_feat_grl.shape
                                valid_s = (torch.arange(Ts, device=self.device)[None, :] < adjusted_lens[:, None]).reshape(-1)
                                src_dom_loss = F_torch.cross_entropy(
                                    src_logits[valid_s],
                                    src_domain_ids.unsqueeze(1).expand(Bs, Ts).reshape(-1)[valid_s],
                                    reduction="mean",
                                )
                            Bt = tgt_feat_grl.shape[0]
                            tgt_logits = self.domain_disc(tgt_feat_grl.float())
                            if mean_pool_for_disc:
                                tgt_dom_loss = F_torch.cross_entropy(
                                    tgt_logits,
                                    torch.full((Bt,), self.n_domains, dtype=torch.long, device=self.device),
                                    reduction="mean",
                                )
                            else:
                                Tt = tgt_feat_grl.shape[1]
                                valid_t = (torch.arange(Tt, device=self.device)[None, :] < adjusted_lens_tgt[:, None]).reshape(-1)
                                tgt_dom_loss = F_torch.cross_entropy(
                                    tgt_logits[valid_t],
                                    torch.full((Bt * Tt,), self.n_domains, dtype=torch.long, device=self.device)[valid_t],
                                    reduction="mean",
                                )
                            dom_loss = 0.5 * (src_dom_loss + lambda_tgt * tgt_dom_loss)

                        if use_two_step:
                            (current_lambda_domain * dom_loss).backward()
                        # (non-two-step combined backward done below)

                # ---------- backward ----------
                if not self.use_mdan:
                    task_loss.backward()
                elif not (self.use_mdan and bool(self.args.get("mdan_two_step_backward", True))):
                    # Non-two-step MDAN: single backward with combined loss
                    combined = task_loss + current_lambda_domain * dom_loss
                    combined.backward()

                grad_norm = 0.0
                if self.args["grad_norm_clip_value"] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.args["grad_norm_clip_value"],
                        error_if_nonfinite=True,
                        foreach=True,
                    )

                self.optimizer.step()
                if self.disc_optimizer is not None:
                    self.disc_optimizer.step()
                self.learning_rate_scheduler.step()

                step_dur = time.time() - start
                loss_for_log = (task_loss.detach() + current_lambda_domain * dom_loss.detach()) if (self.use_mdan and current_lambda_domain > 0) else task_loss.detach()
                if self.rank == 0:
                    train_losses.append(loss_for_log.item())

                if self.rank == 0 and total_steps % self.args["batches_per_train_log"] == 0:
                    self.logger.info(
                        f"Train batch {total_steps}: loss: {loss_for_log.item():.2f} "
                        f"grad norm: {grad_norm:.2f} time: {step_dur:.3f}"
                    )
                    if self.wandb_initialized and wandb.run is not None:
                        cur_lr = self.optimizer.param_groups[0]["lr"]
                        log_dict = {
                            "train/ctc_loss": task_loss.detach().item(),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/learning_rate": cur_lr,
                            "train/time_per_batch": step_dur,
                            "batch": total_steps,
                        }
                        if self.use_mdan:
                            log_dict["train/domain_loss"] = dom_loss.detach().item()
                            log_dict["train/mdan_alpha"] = alpha if self.use_mdan else 0.0
                            log_dict["train/total_loss"] = loss_for_log.item()
                        wandb.log(log_dict, step=total_steps)

                if self.rank == 0 and total_steps > 0 and (
                    total_steps % self.args["batches_per_val_step"] == 0
                    or total_steps == num_training_batches - 1
                ):
                    self.logger.info(f"Running validation after training batch: {total_steps}")
                    vstart = time.time()
                    val_metrics = self.validation(
                        loader=self.val_loader,
                        return_logits=self.args["save_val_logits"],
                        return_data=self.args["save_val_data"],
                    )
                    vdur = time.time() - vstart

                    self.logger.info(
                        f"Val batch {total_steps}: PER (avg): {val_metrics['avg_PER']:.4f} "
                        f"CTC Loss (avg): {val_metrics['avg_loss']:.4f} time: {vdur:.3f}"
                    )
                    if self.args["log_individual_day_val_PER"]:
                        for day in val_metrics["day_PERs"].keys():
                            d = val_metrics["day_PERs"][day]
                            per = (d["total_edit_distance"] / d["total_seq_length"]) if d["total_seq_length"] else 0.0
                            self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {per:0.4f}")

                    if self.wandb_initialized and wandb.run is not None:
                        log = {
                            "eval/ctc_loss": val_metrics["avg_loss"],
                            "eval/cer": val_metrics["avg_PER"],
                            "eval/time": vdur,
                            "batch": total_steps,
                        }
                        if self.args["log_individual_day_val_PER"]:
                            for day in val_metrics["day_PERs"].keys():
                                d = val_metrics["day_PERs"][day]
                                day_per = (d["total_edit_distance"] / d["total_seq_length"]) if d["total_seq_length"] else 0.0
                                log[f"eval/cer_day_{self.args['dataset']['sessions'][day]}"] = day_per
                        wandb.log(log, step=total_steps)

                    val_PERs.append(val_metrics["avg_PER"])
                    val_losses.append(val_metrics["avg_loss"])
                    val_results.append(val_metrics)

                    new_best = False
                    if val_metrics["avg_PER"] < self.best_val_PER:
                        self.logger.info(f"New best PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                        self.best_val_PER = val_metrics["avg_PER"]
                        self.best_val_loss = val_metrics["avg_loss"]
                        new_best = True
                    elif val_metrics["avg_PER"] == self.best_val_PER and val_metrics["avg_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["avg_loss"]
                        new_best = True

                    if new_best:
                        if save_best:
                            self.save_model_checkpoint(
                                f"{self.args['checkpoint_dir']}/best_checkpoint",
                                self.best_val_PER, self.best_val_loss,
                            )
                        if self.args["save_val_metrics"]:
                            with open(f"{self.args['checkpoint_dir']}/val_metrics.pkl", "wb") as f:
                                pickle.dump(val_metrics, f)
                        val_steps_since_improvement = 0
                    else:
                        val_steps_since_improvement += 1

                    if self.args["save_all_val_steps"]:
                        self.save_model_checkpoint(
                            f"{self.args['checkpoint_dir']}/checkpoint_batch_{total_steps}",
                            val_metrics["avg_PER"], val_metrics["avg_loss"],
                        )
                    if early_stopping and val_steps_since_improvement >= early_stopping_val_steps:
                        self.logger.info(f"Early stopping at batch {total_steps}")
                        early_stop_flag = True

                total_steps += 1
                last_i = total_steps
                if total_steps >= num_training_batches:
                    break

            if self.distributed:
                stop_tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
                if self.rank == 0:
                    stop_tensor.fill_(1 if (early_stop_flag or total_steps >= num_training_batches) else 0)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    early_stop_flag = True
                dist.barrier()
            if total_steps >= num_training_batches or early_stop_flag:
                break

        duration = time.time() - train_start
        if self.rank == 0:
            self.logger.info(f"Best avg val PER achieved: {self.best_val_PER:.5f}")
            self.logger.info(f"Total training time: {(duration / 60):.2f} minutes")
            if self.args["save_final_model"] and val_PERs:
                self.save_model_checkpoint(
                    f"{self.args['checkpoint_dir']}/final_checkpoint_batch_{last_i}",
                    val_PERs[-1],
                    val_losses[-1] if val_losses else float("nan"),
                )

        stats = {"train_losses": train_losses, "val_losses": val_losses, "val_PERs": val_PERs, "val_metrics": val_results}
        if self.rank == 0 and self.wandb_initialized and wandb.run is not None:
            wandb.finish()
        if self.distributed:
            dist.barrier()
        return stats

    # -------------------------------------------------------------- val
    def validation(self, loader, return_logits=False, return_data=False):
        self.model.eval()
        metrics = {}
        if return_logits:
            metrics["logits"] = []
            metrics["n_time_steps"] = []
        if return_data:
            metrics["input_features"] = []
        metrics["decoded_seqs"] = []
        metrics["true_seq"] = []
        metrics["phone_seq_lens"] = []
        metrics["transcription"] = []
        metrics["losses"] = []
        metrics["block_nums"] = []
        metrics["trial_nums"] = []
        metrics["day_indicies"] = []

        total_edit = 0
        total_len = 0
        day_per = {}

        for i, batch in enumerate(loader):
            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            phone_seq_lens = batch["phone_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            day = day_indicies[0].item()
            if self.args["dataset"]["dataset_probability_val"][day] == 0:
                if self.args["log_val_skip_logs"]:
                    self.logger.info(f"Skipping validation on day {day}")
                continue

            with torch.no_grad():
                with torch.autocast(device_type=self.autocast_device_type, enabled=self.args["use_amp"], dtype=self.amp_dtype):
                    features, n_time_steps = self.transform_data(features, n_time_steps, "val")
                    adjusted_lens = self._adjusted_lens(n_time_steps)
                    logits = self.model(features, n_time_steps, day_indicies)

                # CTC must run in float32 for numerical stability (outside autocast).
                n_violations = int((adjusted_lens < phone_seq_lens).sum().item())
                if n_violations:
                    self.logger.warning(
                        f"CTC length violation: {n_violations}/{len(adjusted_lens)} samples "
                        f"(min input_len={adjusted_lens.min().item()}, max target_len={phone_seq_lens.max().item()})"
                    )
                loss = self.ctc_loss(
                    torch.permute(logits.float().log_softmax(2), [1, 0, 2]),
                    labels, adjusted_lens, phone_seq_lens,
                )
                loss = torch.mean(loss)

                batch_edit = 0
                decoded_seqs = []
                for it in range(logits.shape[0]):
                    decoded = torch.argmax(logits[it, 0:adjusted_lens[it], :].clone().detach(), dim=-1)
                    decoded = torch.unique_consecutive(decoded, dim=-1)
                    decoded = decoded.cpu().detach().numpy()
                    decoded = np.array([k for k in decoded if k != 0])
                    true = np.array(labels[it][0:phone_seq_lens[it]].cpu().detach())
                    batch_edit += F.edit_distance(decoded, true)
                    decoded_seqs.append(decoded)

            day = batch["day_indicies"][0].item()
            if day not in day_per:
                day_per[day] = {"total_edit_distance": 0, "total_seq_length": 0}
            day_per[day]["total_edit_distance"] += batch_edit
            day_per[day]["total_seq_length"] += torch.sum(phone_seq_lens).item()
            total_edit += batch_edit
            total_len += torch.sum(phone_seq_lens)

            if return_logits:
                metrics["logits"].append(logits.cpu().float().numpy())
                metrics["n_time_steps"].append(adjusted_lens.cpu().numpy())
            if return_data:
                metrics["input_features"].append(batch["input_features"].cpu().numpy())
            metrics["decoded_seqs"].append(decoded_seqs)
            metrics["true_seq"].append(batch["seq_class_ids"].cpu().numpy())
            metrics["phone_seq_lens"].append(batch["phone_seq_lens"].cpu().numpy())
            metrics["transcription"].append(batch["transcriptions"].cpu().numpy())
            metrics["losses"].append(loss.detach().item())
            metrics["block_nums"].append(batch["block_nums"].numpy())
            metrics["trial_nums"].append(batch["trial_nums"].numpy())
            metrics["day_indicies"].append(batch["day_indicies"].cpu().numpy())

        sl = total_len.item() if hasattr(total_len, "item") else total_len
        avg_PER = (total_edit / total_len) if sl else 0.0
        if hasattr(avg_PER, "item"):
            avg_PER = avg_PER.item()

        metrics["day_PERs"] = day_per
        metrics["avg_PER"] = float(avg_PER)
        metrics["avg_loss"] = np.mean(metrics["losses"])
        return metrics
