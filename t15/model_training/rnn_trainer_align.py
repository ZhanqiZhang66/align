# cd /victoriapvc/repos/brain2text-t15/model_training && WANDB_MODE=disabled python test_baseline_vs_mdan_equivalence.py

import torch
import torch.distributed as dist
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle

from dataset import (
    BrainToTextDataset,
    train_test_split_indicies,
    BrainToTextTrialDataset,
    StretchSqueezeDataset,
    generate_prolonged_samples,
    ProlongedDataset,
    collate_trial_batches,
)
from data_augmentations import gauss_smooth
from dann import DomainDiscriminator, grad_reverse, masked_mean_pool, randomly_mask_channelsteps

import torchaudio.functional as F  # for edit distance
from omegaconf import OmegaConf

import wandb

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True
torch._dynamo.config.cache_size_limit = 64

from rnn_model_align import GRUDecoder


def dann_alpha(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None) -> float:
    """Standard DANN schedule: alpha goes from ~0 -> 1 over training."""
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def dann_alpha_alt(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None, phase: int = 16) -> float:
    """Alternative DANN schedule (sinusoidal)."""
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    return float(0.5 + 0.5 * math.sin(phase * math.pi * p - math.pi / 2))

class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder
    
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.distributed = bool(self.args.get('world_size', 1) > 1)
        self.rank = int(self.args.get('rank', 0))
        self.world_size = int(self.args.get('world_size', 1))
        self.local_rank = int(self.args.get('local_rank', 0))

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)

        # Set checkpoint_dir to output_dir if not explicitly provided (null or missing)
        checkpoint_dir = self.args.get('checkpoint_dir')
        if checkpoint_dir is None or (isinstance(checkpoint_dir, str) and checkpoint_dir.strip() == ''):
            self.args['checkpoint_dir'] = self.args['output_dir']

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # Initialize WandB (only if mode is train and WANDB_MODE is not disabled)
        self.wandb_initialized = False
        if args['mode'] == 'train':
            wandb_mode = os.environ.get('WANDB_MODE', 'online').lower()
            if wandb_mode != 'disabled':
                # Get wandb config from args with fallback defaults
                wandb_project = self.args.get('wandb_project', "Brain-to-Text RNN")
                wandb_entity = self.args.get('wandb_entity', "victoriazhang-projects")
                wandb_name = self.args.get('wandb_name', self.args.get('output_dir', 'rnn_training').split('/')[-1])
                
                # Try to login first if API key is available
                wandb_api_key = os.environ.get('WANDB_API_KEY')
                if wandb_api_key:
                    try:
                        wandb.login(key=wandb_api_key, relogin=True)
                        self.logger.info("✅ WandB login successful")
                    except Exception as e:
                        self.logger.warning(f"⚠️  WandB login warning: {e}")
                else:
                    self.logger.warning("⚠️  WANDB_API_KEY not found in environment")
                
                # Prepare wandb init kwargs - ALWAYS online, never offline
                wandb_init_kwargs = {
                    'project': wandb_project,
                    'config': OmegaConf.to_container(args, resolve=True),  # Convert OmegaConf to dict
                    'name': wandb_name,
                    'mode': 'online'  # ALWAYS online - never offline
                }
                
                # Only add entity if it's not None
                if wandb_entity is not None:
                    wandb_init_kwargs['entity'] = wandb_entity
                
                # Handle resume if specified
                if self.args.get('wandb_id') and len(self.args.get('wandb_id', '')) > 0:
                    wandb_init_kwargs['resume'] = "must"
                    wandb_init_kwargs['id'] = self.args['wandb_id']
                
                # Initialize WandB
                try:
                    wandb.init(**wandb_init_kwargs)
                    if wandb.run is not None and wandb.run.url is not None:
                        self.wandb_initialized = True
                        self.logger.info("✅ WandB initialized in ONLINE mode")
                        self.logger.info(f"   Project: {wandb.run.project}")
                        self.logger.info(f"   Entity: {wandb.run.entity}")
                        self.logger.info(f"   Run Name: {wandb.run.name}")
                        self.logger.info(f"   Run URL: {wandb.run.url}")
                    else:
                        raise Exception("WandB run created but no URL (might be offline)")
                except wandb.errors.CommError as e:
                    self.logger.warning(f"⚠️  WandB permission error (bucket creation): {e}")
                    time.sleep(2)
                    if wandb.run is not None and wandb.run.url is not None:
                        self.wandb_initialized = True
                        self.logger.info("   ✅ Run was created despite error - continuing in ONLINE mode")
                    else:
                        self.logger.error("   ❌ Run not created or not online. This may be a permission issue.")
                        self.logger.error("   ❌ WandB ONLINE mode is required - cannot proceed without online logging.")
                        raise RuntimeError("WandB failed to initialize in ONLINE mode. Please check permissions.")
                except Exception as e:
                    self.logger.error(f"❌ WandB initialization error: {e}")
                    time.sleep(1)
                    if wandb.run is None or wandb.run.url is None:
                        self.logger.error("   ❌ WandB ONLINE mode is required - cannot proceed without online logging.")
                        raise RuntimeError("WandB failed to initialize in ONLINE mode.")
                    else:
                        self.wandb_initialized = True
                        self.logger.info("   ✅ WandB run created successfully")
                
            else:
                self.logger.info("⚠️  WandB is disabled via WANDB_MODE environment variable")

        # Configure device: use local_rank when distributed, else gpu_number
        if torch.cuda.is_available():
            if self.distributed:
                gpu_num = self.local_rank
            else:
                gpu_num = self.args.get('gpu_number', 0)
                try:
                    gpu_num = int(gpu_num)
                except ValueError:
                    self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                    gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.rank == 0:
            self.logger.info(f'Using device: {self.device}' + (f' (DDP rank {self.rank}/{self.world_size})' if self.distributed else ''))

        # Per-device batch size: with DDP, use batch_size // world_size so effective batch = config batch_size (avoids OOM)
        cfg_batch = self.args['dataset']['batch_size']
        if self.distributed:
            self.per_device_batch_size = max(1, cfg_batch // self.world_size)
            if self.rank == 0:
                self.logger.info(
                    f"DDP: per-GPU batch_size={self.per_device_batch_size} (config={cfg_batch}, effective={self.per_device_batch_size * self.world_size})"
                )
        else:
            self.per_device_batch_size = cfg_batch

        # Set seed if provided 
        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # Initialize the model 
        self.model = GRUDecoder(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            rnn_dropout = self.args['model']['rnn_dropout'], 
            input_dropout = self.args['model']['input_network']['input_layer_dropout'], 
            n_layers = self.args['model']['n_layers'],
            patch_size = self.args['model']['patch_size'],
            patch_stride = self.args['model']['patch_stride'],
        )

        # Optionally torch.compile to speed up training (can cause dynamo errors with some models)
        if self.args.get("use_torch_compile", False):
            self.logger.info("Using torch.compile")
            self.model = torch.compile(self.model)
        else:
            self.logger.info("Using eager mode (torch.compile disabled; set use_torch_compile: true to enable)")

        self.logger.info(f"Initialized RNN decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        # Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        # Create datasets and dataloaders
        manifest_path = self.args.get('manifest_path')
        # Treat None, missing, empty string, or literal "None" as "no manifest" -> load from dataset_dir
        use_manifest = (
            manifest_path is not None
            and str(manifest_path).strip() not in ("", "None")
            and os.path.exists(str(manifest_path))
        )
        manifest = None
        competition_trials = {}
        if use_manifest:
            # Load from formatted manifest (from format_competition_data_conditions.py)
            self.logger.info(f"Loading datasets from manifest: {manifest_path}")
            with open(manifest_path, 'rb') as f:
                manifest = pickle.load(f)
            training_days = manifest.get('training_days', [])
            target_days = manifest.get('target_days', [])
            test_days = manifest.get('test_days', [])
            self.logger.info(
                f"Manifest split: {len(training_days)} train, {len(target_days)} val/target, {len(test_days)} test (held out for eval)"
            )
            train_trials_raw = manifest.get('train_trial_indicies', {})
            val_trials_raw = manifest.get('val_trial_indicies', {})
            competition_trials_raw = manifest.get('competition_trial_indicies', {})
            
            # Remap relative day indices to absolute session indices
            # Manifest uses relative indices (0..4 for train, 0..12 for target), but model needs absolute indices
            train_trials = {}
            for rel_idx, trial_data in train_trials_raw.items():
                if rel_idx < len(training_days):
                    # Find absolute index in original sessions list
                    session_name = training_days[rel_idx]
                    abs_idx = self.args['dataset']['sessions'].index(session_name) if session_name in self.args['dataset']['sessions'] else rel_idx
                    train_trials[abs_idx] = trial_data
            
            val_trials = {}
            for rel_idx, trial_data in val_trials_raw.items():
                if rel_idx < len(target_days):
                    session_name = target_days[rel_idx]
                    abs_idx = self.args['dataset']['sessions'].index(session_name) if session_name in self.args['dataset']['sessions'] else rel_idx + len(training_days)
                    val_trials[abs_idx] = trial_data
            
            competition_trials = {}
            for rel_idx, trial_data in competition_trials_raw.items():
                if rel_idx < len(target_days):
                    session_name = target_days[rel_idx]
                    abs_idx = self.args['dataset']['sessions'].index(session_name) if session_name in self.args['dataset']['sessions'] else rel_idx + len(training_days)
                    competition_trials[abs_idx] = trial_data
            
            self.logger.info(f"Loaded from manifest: {len(train_trials)} train sessions, {len(val_trials)} val sessions, {len(competition_trials)} competition sessions")
        else:
            # Original logic: load from dataset_dir
            self.logger.info("Creating datasets (loading train/val trial indices from HDF5; may take a while with many sessions)...")
            train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
            val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

            # Ensure that there are no duplicate days
            if len(set(train_file_paths)) != len(train_file_paths):
                raise ValueError("There are duplicate sessions listed in the train dataset")
            if len(set(val_file_paths)) != len(val_file_paths):
                raise ValueError("There are duplicate sessions listed in the val dataset")

            # Split trials into train and test sets
            train_trials, _ = train_test_split_indicies(
                file_paths = train_file_paths, 
                test_percentage = 0,
                seed = self.args['dataset']['seed'],
                bad_trials_dict = None,
                )
            _, val_trials = train_test_split_indicies(
                file_paths = val_file_paths, 
                test_percentage = 1,
                seed = self.args['dataset']['seed'],
                bad_trials_dict = None,
                )
            competition_trials = {}  # Not available in old mode
            n_train = sum(len(v['trials']) for v in train_trials.values())
            n_val = sum(len(v['trials']) for v in val_trials.values())
            self.logger.info(f"Loaded from dataset_dir: {len(train_trials)} train sessions ({n_train} trials), {len(val_trials)} val sessions ({n_val} trials)")

        self.logger.info("Building train/val Dataset and DataLoader objects...")
        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials, 'competition': competition_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.per_device_batch_size,
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        train_sampler = None
        train_shuffle = self.args['dataset']['loader_shuffle']
        if self.distributed:
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            train_shuffle = False
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.args['dataset']['num_dataloader_workers'],
            pin_memory=True
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None,
            shuffle = False,
            num_workers = 0,
            pin_memory = True
        )

        # ---------- MDAN/DANN: source/target split and domain discriminator ----------
        self.use_mdan = (
            self.args.get('source_days') is not None
            and self.args.get('target_days') is not None
            and len(self.args.get('source_days', [])) > 0
            and len(self.args.get('target_days', [])) > 0
        )
        self.train_loader_source = None
        self.train_loader_target = None
        self.domain_disc = None
        self.disc_optimizer = None
        self.day_index_to_domain_id = None
        self.n_domains = 0
        self.truely_mdan = False

        if self.use_mdan:
            source_days = list(self.args['source_days'])
            target_days = list(self.args['target_days'])
            self.n_domains = len(source_days)
            self.truely_mdan = bool(self.args.get('truely_mdan', True))
            rep_layer_idx = self.args.get('rep_layer_idx', None)
            # Representation dimension: input_size for -1 (before GRU), n_units for GRU layers (0 or positive)
            rep_dim = self.model.input_size if rep_layer_idx == -1 else self.model.n_units
            lambda_domain = float(self.args.get('dann_lambda', 0.001))
            domain_hidden = int(self.args.get('dann_hidden', 256))
            domain_dropout = float(self.args.get('domain_dropout', 0.1))
            linear_discriminator = bool(self.args.get('linear_discriminator', True))
            dann_lr_multiplier = float(self.args.get('dann_lr_multiplier', 1.0))
            dann_weight_decay = self.args.get('dann_weight_decay', self.args['weight_decay'])

            # If manifest_path provided, use competition_trials for target; else fallback to train_trials
            if manifest and len(competition_trials) > 0:
                # Use formatted data: source = train, target = competition (from test files)
                # train_trials and competition_trials now use absolute session indices (after remapping)
                train_trials_source = {d: train_trials[d] for d in source_days if d in train_trials and len(train_trials[d]['trials']) > 0}
                train_trials_target = {d: competition_trials[d] for d in target_days if d in competition_trials and len(competition_trials[d]['trials']) > 0}
                self.logger.info(f"MDAN: Using formatted data - source from train ({len(train_trials_source)} sessions), target from competition ({len(train_trials_target)} sessions)")
            else:
                # Fallback: use train_trials for both (old behavior)
                train_trials_source = {d: train_trials[d] for d in source_days if d in train_trials and len(train_trials[d]['trials']) > 0}
                train_trials_target = {d: train_trials[d] for d in target_days if d in train_trials and len(train_trials[d]['trials']) > 0}
                self.logger.info(f"MDAN: Using fallback - source from train ({len(train_trials_source)} sessions), target from train ({len(train_trials_target)} sessions)")
            
            if len(train_trials_source) == 0 or len(train_trials_target) == 0:
                raise ValueError("MDAN: source_days or target_days yielded no training trials. Check source_days/target_days match session indices.")
            # Map day index (in batch) -> domain id 0..K-1 for source days
            self.day_index_to_domain_id = {d: idx for idx, d in enumerate(source_days)}

            # Optional stretch/squeeze and prolonged augmentations (trial-level datasets)
            include_stretched = self.args.get('include_stretched_samples', False)
            include_prolonged = self.args.get('include_prolonged_samples', False)
            include_original = self.args.get('include_original', True)
            stretch_range = self.args.get('stretch_range', 2.0)
            prolonged_combined_range = tuple(self.args.get('prolonged_combined_range', [1, 10]))
            prolonged_sample_size = self.args.get('prolonged_sample_size')

            use_trial_level_source = include_stretched or include_prolonged
            if use_trial_level_source:
                base_src_trial_ds = BrainToTextTrialDataset(train_trials_source, feature_subset=feature_subset)
                src_components = []
                if include_stretched:
                    src_components.append(StretchSqueezeDataset(base_src_trial_ds, stretch_range=stretch_range))
                if include_prolonged:
                    prolonged_list = generate_prolonged_samples(
                        base_src_trial_ds,
                        combined_range=prolonged_combined_range,
                        stretch_range=stretch_range,
                        sample_size=prolonged_sample_size,
                    )
                    src_components.append(ProlongedDataset(prolonged_list))
                if include_original:
                    src_components.append(base_src_trial_ds)
                self.train_dataset_source = ConcatDataset(src_components)
                if self.rank == 0:
                    self.logger.info(f"MDAN source: trial-level aug (stretch={include_stretched}, prolonged={include_prolonged}, original={include_original}), len={len(self.train_dataset_source)}")
            else:
                self.train_dataset_source = BrainToTextDataset(
                    trial_indicies=train_trials_source,
                    split='train',
                    days_per_batch=min(self.args['dataset']['days_per_batch'], len(source_days)),
                    n_batches=self.args['num_training_batches'],
                    batch_size=self.per_device_batch_size,
                    must_include_days=None,
                    random_seed=self.args['dataset']['seed'],
                    feature_subset=feature_subset
                )

            src_sampler = None
            src_shuffle = self.args['dataset']['loader_shuffle']
            if self.distributed:
                src_sampler = DistributedSampler(self.train_dataset_source, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                src_shuffle = False
            self.train_loader_source = DataLoader(
                self.train_dataset_source,
                batch_size=self.per_device_batch_size if use_trial_level_source else None,
                shuffle=src_shuffle,
                sampler=src_sampler,
                num_workers=self.args['dataset']['num_dataloader_workers'],
                pin_memory=True,
                collate_fn=collate_trial_batches if use_trial_level_source else None,
            )

            # Target: optional stretch augmentation (trial-level when stretch enabled)
            n_batches_target = max(1, self.args['num_training_batches'] // 2)
            if include_stretched:
                base_tgt_trial_ds = BrainToTextTrialDataset(train_trials_target, feature_subset=feature_subset)
                stretched_tgt = StretchSqueezeDataset(base_tgt_trial_ds, stretch_range=stretch_range)
                self.train_dataset_target = ConcatDataset([base_tgt_trial_ds, stretched_tgt]) if include_original else stretched_tgt
                if self.rank == 0:
                    self.logger.info(f"MDAN target: trial-level + stretch (original={include_original}), len={len(self.train_dataset_target)}")
            else:
                self.train_dataset_target = BrainToTextDataset(
                    trial_indicies=train_trials_target,
                    split='train',
                    days_per_batch=min(self.args['dataset']['days_per_batch'], len(target_days)),
                    n_batches=n_batches_target,
                    batch_size=self.per_device_batch_size,
                    must_include_days=None,
                    random_seed=self.args['dataset']['seed'],
                    feature_subset=feature_subset
                )

            tgt_sampler = None
            tgt_shuffle = True
            if self.distributed:
                tgt_sampler = DistributedSampler(self.train_dataset_target, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                tgt_shuffle = False
            self.train_loader_target = DataLoader(
                self.train_dataset_target,
                batch_size=self.per_device_batch_size if include_stretched else None,
                shuffle=tgt_shuffle,
                sampler=tgt_sampler,
                num_workers=self.args['dataset']['num_dataloader_workers'],
                pin_memory=True,
                collate_fn=collate_trial_batches if include_stretched else None,
            )

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
            disc_lr = self.args['lr_max'] * dann_lr_multiplier
            disc_wd = dann_weight_decay if dann_weight_decay is not None else self.args['weight_decay']
            use_fused = self.device.type in ('cuda', 'xpu', 'privateuseone')
            self.disc_optimizer = torch.optim.AdamW(
                disc_params,
                lr=disc_lr,
                weight_decay=disc_wd,
                betas=(self.args['beta0'], self.args['beta1']),
                fused=use_fused
            )
            self.logger.info(
                f"MDAN enabled: source_days={source_days}, target_days={target_days}, "
                f"n_domains={self.n_domains}, truely_mdan={self.truely_mdan}, rep_dim={rep_dim}"
            )

        self.logger.info("Successfully initialized datasets")

        # Set rnn and/or input layers to not trainable if specified 
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device before creating optimizer (required for fused AdamW on CUDA)
        self.model.to(self.device)
        if self.distributed:
            # find_unused_parameters=True required: two-step backward and/or per-GPU batches leave
            # some params (e.g. target-day layers) unused in the first backward
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        # Scale scheduler length by world_size so each rank's schedule matches its step count (same total effective batches as single GPU).
        effective_lr_decay_steps = self.args['lr_decay_steps'] // self.world_size
        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = effective_lr_decay_steps,
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        # If a checkpoint is provided, load it (after optimizer/scheduler exist)
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Watch the model if wandb is initialized (after model is on device).
        # Do not use log="all" / gradient hooks: MDAN uses multiple backward passes and some
        # tensors get None grads; wandb's hooks then crash (AttributeError on grad.data).
        if self.wandb_initialized and wandb.run is not None:
            model_for_watch = self.model.module if self.distributed else self.model
            wandb.watch(model_for_watch, log="parameters")
            self.logger.info("✅ WandB watching model parameters (gradients disabled for MDAN compatibility)")

    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 

        Biases and day weights should not be decayed

        Day weights should have a separate learning rate
        '''
        def is_bias(name):
            return 'out.bias' in name or 'gru.bias' in name or ('gru_layers' in name and 'bias' in name)
        bias_params = [p for name, p in self.model.named_parameters() if is_bias(name)]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and not is_bias(name)]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
            
        # Fused AdamW only supports CUDA (and xpu/privateuseone); use False on CPU
        use_fused = self.device.type in ('cuda', 'xpu', 'privateuseone')
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = use_fused
        )

        return optim 

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        # Per-rank steps so schedule matches steps_per_rank (same total effective batches as single GPU).
        lr_decay_steps = self.args['lr_decay_steps'] // self.world_size

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day'] // self.world_size

        lr_warmup_steps = self.args['lr_warmup_steps'] // self.world_size
        lr_warmup_steps_day = self.args['lr_warmup_steps_day'] // self.world_size

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay

            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        '''
        Load a training checkpoint. load_path can be a file (e.g. best_checkpoint) or a
        directory containing best_checkpoint (we load that file).
        '''
        load_path = os.path.abspath(load_path)
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, 'best_checkpoint')
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint path is not a file: {load_path}")
        checkpoint = torch.load(load_path, weights_only=False)
        model_to_load = self.model.module if self.distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER']
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        if self.domain_disc is not None and 'domain_disc_state_dict' in checkpoint:
            self.domain_disc.load_state_dict(checkpoint['domain_disc_state_dict'])
        if self.disc_optimizer is not None and 'disc_optimizer_state_dict' in checkpoint:
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

        self.model.to(self.device)
        if self.domain_disc is not None:
            self.domain_disc.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        if self.disc_optimizer is not None:
            for state in self.disc_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if self.rank == 0:
            self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint (saves on rank 0 only when DDP).
        '''
        model_to_save = self.model.module if self.distributed else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': PER,
            'val_loss': loss
        }
        if self.domain_disc is not None:
            checkpoint['domain_disc_state_dict'] = self.domain_disc.state_dict()
        if self.disc_optimizer is not None:
            checkpoint['disc_optimizer_state_dict'] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, save_path)

        if self.rank == 0:
            self.logger.info("Saved model to checkpoint: " + save_path)
            with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
                OmegaConf.save(config=self.args, f=f)

    def create_attention_mask(self, sequence_lengths):

        max_length = torch.max(sequence_lengths).item()

        batch_size = sequence_lengths.size(0)
        
        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)
        
        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)
        
        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)
        
        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask, 
                                        True,
                                        False)
        
        return attention_mask_float

    def transform_data(self, features, n_time_steps, mode = 'train'):
        '''
        Apply various augmentations and smoothing to data
        Performing augmentations is much faster on GPU than CPU
        '''

        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        
        return features, n_time_steps

    def train(self):
        '''
        Train the model 
        '''

        # Set model to train mode (specificially to make sure dropout layers are engaged)
        self.model.train()

        # create vars to track performance
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        # training params 
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)

        early_stopping_val_steps = self.args['early_stopping_val_steps']

        train_start_time = time.time()
        num_training_batches = self.args['num_training_batches']
        # Total effective batches = num_training_batches. With DDP, each rank runs steps_per_rank
        # so that steps_per_rank * world_size == num_training_batches (same total data as single GPU).
        steps_per_rank = num_training_batches // self.world_size

        # Run enough epochs to reach steps_per_rank steps on this rank.
        # With DDP, each rank sees len(loader) = dataset_size // world_size per epoch.
        train_loop_loader = self.train_loader_source if self.use_mdan else self.train_loader
        steps_per_epoch = len(train_loop_loader)
        num_epochs = max(1, (steps_per_rank + steps_per_epoch - 1) // steps_per_epoch) if steps_per_epoch > 0 else 1

        target_iter = iter(self.train_loader_target) if self.use_mdan else None
        rep_layer_idx = self.args.get('rep_layer_idx', None)
        lambda_domain = float(self.args.get('dann_lambda', 0.001))
        dann_warmup_steps = int(self.args.get('dann_warmup_steps', 0))
        _alpha_max = self.args.get('dann_alpha_max_steps')
        alpha_max_steps = int(_alpha_max if _alpha_max is not None else steps_per_rank)
        mean_pool_for_disc = bool(self.args.get('mean_pool_for_discriminator', True))
        lambda_tgt = float(self.args.get('lambda_tgt', 1.0))
        dom_loss_type = self.args.get('dom_loss_type', 'mean')

        total_steps = 0
        last_i = 0
        early_stop_flag = False

        for epoch in range(num_epochs):
            if self.distributed and hasattr(train_loop_loader, 'sampler') and hasattr(train_loop_loader.sampler, 'set_epoch'):
                train_loop_loader.sampler.set_epoch(epoch)
            if self.use_mdan and self.distributed and hasattr(self.train_loader_target, 'sampler') and hasattr(self.train_loader_target.sampler, 'set_epoch'):
                self.train_loader_target.sampler.set_epoch(epoch)
            target_iter = iter(self.train_loader_target) if self.use_mdan else None

            for i, batch in enumerate(train_loop_loader):
                self.model.train()
                if self.domain_disc is not None:
                    self.domain_disc.train()
                self.optimizer.zero_grad(set_to_none=True)
                if self.disc_optimizer is not None:
                    self.disc_optimizer.zero_grad(set_to_none=True)

                start_time = time.time()

                # ---------- Source batch ----------
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)

                with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'train')
                    unwrapped = self.model.module if self.distributed else self.model
                    adjusted_lens = unwrapped.compute_length(n_time_steps)

                    if self.use_mdan:
                        logits, src_rep = self.model(features, day_indicies, return_rep=True, rep_layer_idx=rep_layer_idx)
                    else:
                        logits = self.model(features, day_indicies)

                    task_loss = self.ctc_loss(
                        log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        targets=labels,
                        input_lengths=adjusted_lens,
                        target_lengths=phone_seq_lens
                    )
                    task_loss = torch.mean(task_loss)

                use_two_step_backward = False
                if self.use_mdan:
                    # Two-step backward to avoid OOM: backward task loss first (frees source graph),
                    # then target forward + domain backward. Domain loss uses detached src_rep so we
                    # never hold both full forward graphs. Enables batch_size 64 on ~22GB GPUs.
                    use_two_step_backward = bool(self.args.get('mdan_two_step_backward', True))
                    if use_two_step_backward:
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.disc_optimizer is not None:
                            self.disc_optimizer.zero_grad(set_to_none=True)
                        task_loss.backward()
                        # Keep minimal source-side info for domain loss (detach so no second graph)
                        adjusted_lens_src = adjusted_lens  # keep for domain loss masking
                        if self.args.get('dann_dropout_channel_prob', 0.0) > 0:
                            src_rep_det = randomly_mask_channelsteps(src_rep.detach(), adjusted_lens_src, self.args['dann_dropout_channel_prob'])
                        else:
                            src_rep_det = src_rep.detach()
                        if mean_pool_for_disc:
                            src_feat_det = masked_mean_pool(src_rep_det, adjusted_lens_src)
                        else:
                            src_feat_det = src_rep_det
                        src_domain_ids = torch.tensor(
                            [self.day_index_to_domain_id[int(d)] for d in day_indicies.cpu().tolist()],
                            dtype=torch.long, device=self.device
                        )
                        del features, logits, src_rep
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # Alpha for GRL (single backward)
                        warmup_steps = dann_warmup_steps
                        alpha_total = max(steps_per_rank - warmup_steps, 1)
                        alpha_step = max(total_steps - warmup_steps, 0)
                        is_warmup = total_steps < warmup_steps
                        if is_warmup:
                            alpha = 0.0
                        else:
                            if self.args.get('dann_alpha_type', 'standard') == 'alternative':
                                alpha = dann_alpha_alt(alpha_step, alpha_total, alpha_max_steps=alpha_max_steps)
                            else:
                                alpha = dann_alpha(alpha_step, alpha_total, alpha_max_steps=alpha_max_steps)
                        current_lambda_domain = 0.0 if is_warmup else lambda_domain
                        if self.args.get('dann_dropout_channel_prob', 0.0) > 0:
                            src_rep = randomly_mask_channelsteps(src_rep, adjusted_lens, self.args['dann_dropout_channel_prob'])
                        if mean_pool_for_disc:
                            src_feat = masked_mean_pool(src_rep, adjusted_lens)
                        else:
                            src_feat = src_rep
                        src_feat_disc = src_feat
                        if is_warmup:
                            src_feat_grl = src_feat_disc.detach()
                        else:
                            src_feat_grl = grad_reverse(src_feat_disc, alpha=alpha)
                        src_domain_ids = torch.tensor(
                            [self.day_index_to_domain_id[int(d)] for d in day_indicies.cpu().tolist()],
                            dtype=torch.long, device=self.device
                        )

                    if use_two_step_backward:
                        warmup_steps = dann_warmup_steps
                        alpha_total = max(steps_per_rank - warmup_steps, 1)
                        alpha_step = max(total_steps - warmup_steps, 0)
                        is_warmup = total_steps < warmup_steps
                        if is_warmup:
                            alpha = 0.0
                        else:
                            if self.args.get('dann_alpha_type', 'standard') == 'alternative':
                                alpha = dann_alpha_alt(alpha_step, alpha_total, alpha_max_steps=alpha_max_steps)
                            else:
                                alpha = dann_alpha(alpha_step, alpha_total, alpha_max_steps=alpha_max_steps)
                        current_lambda_domain = 0.0 if is_warmup else lambda_domain
                        if is_warmup:
                            src_feat_grl = src_feat_det
                        else:
                            src_feat_grl = grad_reverse(src_feat_det, alpha=alpha)
                    # Target batch
                    try:
                        tgt_batch = next(target_iter)
                    except (StopIteration, TypeError):
                        target_iter = iter(self.train_loader_target)
                        tgt_batch = next(target_iter)
                    tgt_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in tgt_batch.items()}
                    tgt_features = tgt_batch['input_features'].to(self.device)
                    tgt_n_time_steps = tgt_batch['n_time_steps'].to(self.device)
                    with torch.autocast(device_type="cuda", enabled=self.args['use_amp'], dtype=torch.bfloat16):
                        tgt_features, tgt_n_time_steps = self.transform_data(tgt_features, tgt_n_time_steps, 'train')
                        adjusted_lens_tgt = unwrapped.compute_length(tgt_n_time_steps)
                        _, tgt_rep = self.model(tgt_features, tgt_batch['day_indicies'].to(self.device), return_rep=True, rep_layer_idx=rep_layer_idx)
                    if self.args.get('dann_dropout_channel_prob', 0.0) > 0:
                        tgt_rep = randomly_mask_channelsteps(tgt_rep, adjusted_lens_tgt, self.args['dann_dropout_channel_prob'])
                    if mean_pool_for_disc:
                        tgt_feat = masked_mean_pool(tgt_rep, adjusted_lens_tgt)
                    else:
                        tgt_feat = tgt_rep
                    tgt_feat_disc = tgt_feat
                    if is_warmup:
                        tgt_feat_grl = tgt_feat_disc.detach()
                    else:
                        tgt_feat_grl = grad_reverse(tgt_feat_disc, alpha=alpha)

                    def hinge_loss_logits(logits, target_is_source):
                        t = 1.0 if target_is_source else -1.0
                        return torch.clamp(1.0 - t * logits, min=0.0)

                    if (use_two_step_backward and current_lambda_domain > 0) or (not use_two_step_backward and current_lambda_domain > 0) or (not use_two_step_backward and is_warmup):
                        if not use_two_step_backward:
                            current_lambda_domain = 0.0 if is_warmup else lambda_domain
                    if not use_two_step_backward:
                        is_warmup = total_steps < dann_warmup_steps
                        current_lambda_domain = 0.0 if is_warmup else lambda_domain
                    if current_lambda_domain > 0 or (not use_two_step_backward and is_warmup):
                        if self.truely_mdan:
                            dom_losses = []
                            for disc_idx, disc in enumerate(self.domain_disc):
                                src_mask = (src_domain_ids == disc_idx)
                                m = int(src_mask.sum().item())
                                if src_mask.any():
                                    if mean_pool_for_disc:
                                        sel_s = src_mask
                                    else:
                                        Bs, Ts, D = src_feat_grl.shape
                                        t = torch.arange(Ts, device=self.device)[None, :].expand(Bs, Ts)
                                        adj_lens = adjusted_lens_src if use_two_step_backward else adjusted_lens
                                        adj_mask = (t < adj_lens[:, None])
                                        sel_s = src_mask[:, None] & adj_mask
                                    logits_s = disc(src_feat_grl[sel_s].float()).view(-1)
                                    src_loss_i = hinge_loss_logits(logits_s, True).mean()
                                else:
                                    src_loss_i = torch.tensor(0.0, device=self.device)
                                if m > 0 and tgt_feat_grl is not None:
                                    Bt = tgt_feat_grl.shape[0]
                                    mm = min(m, Bt)
                                    tgt_mask = torch.randperm(Bt, device=self.device)[:mm]
                                    if mean_pool_for_disc:
                                        logits_t = disc(tgt_feat_grl[tgt_mask].float()).view(-1)
                                    else:
                                        tt = torch.arange(tgt_feat_grl.shape[1], device=self.device)[None, :].expand(mm, tgt_feat_grl.shape[1])
                                        sel_t = (tt < adjusted_lens_tgt[tgt_mask][:, None])
                                        logits_t = disc(tgt_feat_grl[tgt_mask][sel_t].float()).view(-1)
                                    tgt_loss_i = hinge_loss_logits(logits_t, False).mean()
                                else:
                                    tgt_loss_i = torch.tensor(0.0, device=self.device)
                                dom_losses.append(0.5 * (src_loss_i + lambda_tgt * tgt_loss_i))
                            if len(dom_losses) > 0:
                                dom_stack = torch.stack(dom_losses)
                                if dom_loss_type == 'max':
                                    dom_loss = dom_stack.max()
                                elif dom_loss_type == 'mean':
                                    dom_loss = dom_stack.mean()
                                else:
                                    dom_loss = dom_stack.mean()
                            else:
                                dom_loss = torch.tensor(0.0, device=self.device)
                        else:
                            src_logits = self.domain_disc(src_feat_grl.float())
                            if mean_pool_for_disc:
                                src_dom_loss = F_torch.cross_entropy(src_logits, src_domain_ids, reduction='mean')
                            else:
                                Bs, Ts, _ = src_feat_grl.shape
                                valid_s = (torch.arange(Ts, device=self.device)[None, :] < adjusted_lens[:, None]).reshape(-1)
                                src_dom_loss = F_torch.cross_entropy(
                                    src_logits[valid_s],
                                    src_domain_ids.unsqueeze(1).expand(Bs, Ts).reshape(-1)[valid_s],
                                    reduction='mean'
                                )
                            Bt = tgt_feat_grl.shape[0]
                            tgt_logits = self.domain_disc(tgt_feat_grl.float())
                            if mean_pool_for_disc:
                                tgt_dom_loss = F_torch.cross_entropy(
                                    tgt_logits,
                                    torch.full((Bt,), self.n_domains, dtype=torch.long, device=self.device),
                                    reduction='mean'
                                )
                            else:
                                Tt = tgt_feat_grl.shape[1]
                                valid_t = (torch.arange(Tt, device=self.device)[None, :] < adjusted_lens_tgt[:, None]).reshape(-1)
                                tgt_dom_loss = F_torch.cross_entropy(
                                    tgt_logits[valid_t],
                                    torch.full((Bt * Tt,), self.n_domains, dtype=torch.long, device=self.device)[valid_t],
                                    reduction='mean'
                                )
                            dom_loss = 0.5 * (src_dom_loss + lambda_tgt * tgt_dom_loss)
                        if use_two_step_backward:
                            (current_lambda_domain * dom_loss).backward()
                        else:
                            mdan_loss = task_loss + current_lambda_domain * dom_loss
                            loss = mdan_loss
                    else:
                        if use_two_step_backward:
                            pass  # already did task_loss.backward()
                        else:
                            loss = task_loss
                    if not use_two_step_backward:
                        loss = mdan_loss if (self.use_mdan and current_lambda_domain > 0) else task_loss
                    if use_two_step_backward:
                        loss = (task_loss.detach() + current_lambda_domain * dom_loss.detach()) if current_lambda_domain > 0 else task_loss.detach()
                else:
                    loss = task_loss

                if not (self.use_mdan and use_two_step_backward):
                    loss.backward()

                grad_norm = 0.0
                if self.args['grad_norm_clip_value'] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.args['grad_norm_clip_value'],
                        error_if_nonfinite=True,
                        foreach=True
                    )

                self.optimizer.step()
                if self.disc_optimizer is not None:
                    self.disc_optimizer.step()
                self.learning_rate_scheduler.step()

                train_step_duration = time.time() - start_time
                if self.rank == 0:
                    train_losses.append(loss.detach().item())

                if self.rank == 0 and total_steps % self.args['batches_per_train_log'] == 0:
                    self.logger.info(
                        f'Train batch {total_steps}: loss: {(loss.detach().item()):.2f} '
                        f'grad norm: {grad_norm:.2f} time: {train_step_duration:.3f}'
                    )
                    if self.wandb_initialized and wandb.run is not None:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        log_dict = {
                            "train/ctc_loss": loss.detach().item(),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/learning_rate": current_lr,
                            "train/time_per_batch": train_step_duration,
                            "batch": total_steps
                        }
                        if self.use_mdan:
                            log_dict["train/mdan_loss"] = loss.detach().item()
                            log_dict["train/task_loss"] = task_loss.detach().item()
                        wandb.log(log_dict, step=total_steps)

                if self.rank == 0 and (total_steps % self.args['batches_per_val_step'] == 0 or total_steps == steps_per_rank - 1):
                    self.logger.info(f"Running validation after training batch: {total_steps}")
                    start_time = time.time()
                    val_metrics = self.validation(loader=self.val_loader, return_logits=self.args['save_val_logits'], return_data=self.args['save_val_data'])
                    val_step_duration = time.time() - start_time

                    self.logger.info(
                        f'Val batch {total_steps}: PER (avg): {val_metrics["avg_PER"]:.4f} '
                        f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} time: {val_step_duration:.3f}'
                    )
                    if self.args['log_individual_day_val_PER']:
                        for day in val_metrics['day_PERs'].keys():
                            d = val_metrics['day_PERs'][day]
                            per = (d['total_edit_distance'] / d['total_seq_length']) if d['total_seq_length'] else 0.0
                            self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {per:0.4f}")

                    if self.wandb_initialized and wandb.run is not None:
                        log_dict = {
                            "eval/ctc_loss": val_metrics["avg_loss"],
                            "eval/cer": val_metrics["avg_PER"],
                            "eval/time": val_step_duration,
                            "batch": total_steps
                        }
                        if self.args['log_individual_day_val_PER']:
                            for day in val_metrics['day_PERs'].keys():
                                d = val_metrics['day_PERs'][day]
                                day_per_val = (d['total_edit_distance'] / d['total_seq_length']) if d['total_seq_length'] else 0.0
                                log_dict[f"eval/cer_day_{self.args['dataset']['sessions'][day]}"] = day_per_val
                        wandb.log(log_dict, step=total_steps)

                    val_PERs.append(val_metrics['avg_PER'])
                    val_losses.append(val_metrics['avg_loss'])
                    val_results.append(val_metrics)

                    new_best = False
                    if val_metrics['avg_PER'] < self.best_val_PER:
                        self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                        self.best_val_PER = val_metrics['avg_PER']
                        self.best_val_loss = val_metrics['avg_loss']
                        new_best = True
                    elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss):
                        self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                        self.best_val_loss = val_metrics['avg_loss']
                        new_best = True

                    if new_best:
                        if save_best_checkpoint:
                            self.logger.info("Checkpointing model")
                            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)
                        if self.args['save_val_metrics']:
                            with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                                pickle.dump(val_metrics, f)
                        val_steps_since_improvement = 0
                    else:
                        val_steps_since_improvement += 1

                    if self.args['save_all_val_steps']:
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{total_steps}', val_metrics['avg_PER'])

                    if early_stopping and val_steps_since_improvement >= early_stopping_val_steps:
                        self.logger.info(f'Early stopping at batch {total_steps}')
                        early_stop_flag = True

                total_steps += 1
                last_i = total_steps
                if total_steps >= steps_per_rank:
                    break

                if self.distributed:
                    stop_tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
                    if self.rank == 0:
                        stop_tensor.fill_(1 if (early_stop_flag or total_steps >= steps_per_rank) else 0)
                    dist.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        early_stop_flag = True
                    dist.barrier()
                if total_steps >= steps_per_rank or early_stop_flag:
                    break
            if total_steps >= steps_per_rank or early_stop_flag:
                break

        training_duration = time.time() - train_start_time
        if self.rank == 0:
            self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
            self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')
            if self.args['save_final_model'] and val_PERs:
                self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{last_i}', val_PERs[-1])

        train_stats = {'train_losses': train_losses, 'val_losses': val_losses, 'val_PERs': val_PERs, 'val_metrics': val_results}
        if self.rank == 0 and self.wandb_initialized and wandb.run is not None:
            wandb.finish()
            self.logger.info("✅ WandB run finished")

        if self.distributed:
            dist.barrier()
        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}
        
        # Record metrics
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0
        total_seq_length = 0

        # Calculate PER for each specific day (only for days that appear in the loader)
        # With manifest-based MDAN, val_trials only has target sessions (e.g. 5–17), not train (0–4),
        # so we init lazily to avoid day_per entries with total_seq_length=0 and division-by-zero when logging.
        day_per = {}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # Determine if we should perform validation on this batch
            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)

                    logits = self.model(features, day_indicies)
    
                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)

                metrics['losses'].append(loss.cpu().detach().numpy())

                # Calculate PER per day and also avg over entire validation set
                batch_edit_distance = 0 
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
            
                    batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)

                    decoded_seqs.append(decoded_seq)

            day = batch['day_indicies'][0].item()
            if day not in day_per:
                day_per[day] = {'total_edit_distance': 0, 'total_seq_length': 0}
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics 