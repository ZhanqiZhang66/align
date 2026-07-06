import os
import pickle
import time
import math
from typing import Any, Dict, Optional, Tuple

from edit_distance import SequenceMatcher
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from .dataset import getDatasetLoaders, getDatasetLoadersInterleaved, SpeechDataset, StretchSqueezeDataset
import torch.nn.functional as F
from .loss import forward_ctc, forward_cr_ctc, future_prediction_loss, phone_contrastive_loss, ctc_run_alignment_phone_ids, cross_trial_phone_contrastive_loss, forward_ctc_ntp, weighted_forward_ctc, kl_phone_prior_loss
from .align import DomainDiscriminator, grad_reverse, masked_mean_pool, randomly_mask_channelsteps


import wandb


# -------------------------
# DANN Utilities
# -------------------------
sessionNames = ['t12.2022.04.28',  't12.2022.05.26',  't12.2022.06.21',  't12.2022.07.21',  't12.2022.08.13',
't12.2022.05.05',  't12.2022.06.02',  't12.2022.06.23',  't12.2022.07.27',  't12.2022.08.18',
't12.2022.05.17',  't12.2022.06.07',  't12.2022.06.28',  't12.2022.07.29',  't12.2022.08.23',
't12.2022.05.19',  't12.2022.06.14',  't12.2022.07.05',  't12.2022.08.02',  't12.2022.08.25',
't12.2022.05.24',  't12.2022.06.16',  't12.2022.07.14',  't12.2022.08.11']

sessionNames.remove('t12.2022.07.29') # this was neither used in training nor validation. 
sessionNames.remove('t12.2022.08.18') # this was left for testing. 
sessionNames.remove('t12.2022.08.23') # this was left for testing
sessionNames.remove('t12.2022.08.25') # this was left for testing

sessionNames.sort()

def masked_mean_1d(x_bt: torch.Tensor, lens_b: torch.Tensor) -> torch.Tensor:
    # x_bt: [B, T], lens_b: [B]
    B, T = x_bt.shape
    t = torch.arange(T, device=x_bt.device)[None, :].expand(B, T)
    m = (t < lens_b[:, None]).float()
    return (x_bt * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))

# Phone mapping
PHONE_DEF = [
	'AA', 'AE', 'AH', 'AO', 'AW',
	'AY', 'B',  'CH', 'D', 'DH',
	'EH', 'ER', 'EY', 'F', 'G',
	'HH', 'IH', 'IY', 'JH', 'K',
	'L', 'M', 'N', 'NG', 'OW',
	'OY', 'P', 'R', 'S', 'SH',
	'T', 'TH', 'UH', 'UW', 'V',
	'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']
# Blank token (index 0 in CTC)
BLANK_TOKEN = 0
phoneme_to_id = {"BLANK": BLANK_TOKEN}

for i, p in enumerate(PHONE_DEF_SIL):
    phoneme_to_id[p] = i + 1  # +1 because 0 is CTC blank

id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
VOWEL_PHONEMES = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW",
]
VOWEL_IDS = {phoneme_to_id[p] for p in VOWEL_PHONEMES if p in phoneme_to_id}
CONSONANT_IDS = {
    phoneme_to_id[p] for p in PHONE_DEF if p not in VOWEL_PHONEMES
}

def _update_cer_breakdown(
    true_seq: np.ndarray,
    pred_seq: np.ndarray,
    per_phone_err: np.ndarray,
    per_phone_count: np.ndarray,
    vowel_err: int,
    vowel_count: int,
    consonant_err: int,
    consonant_count: int,
):
    matcher = SequenceMatcher(a=true_seq.tolist(), b=pred_seq.tolist())
    opcodes = matcher.get_opcodes()

    for p in true_seq:
        per_phone_count[p] += 1
        if p in VOWEL_IDS:
            vowel_count += 1
        elif p in CONSONANT_IDS:
            consonant_count += 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        true_seg = true_seq[i1:i2]
        pred_seg = pred_seq[j1:j2]

        if tag == "replace":
            min_len = min(len(true_seg), len(pred_seg))
            # substitutions
            for p in true_seg[:min_len]:
                per_phone_err[p] += 1
                if p in VOWEL_IDS:
                    vowel_err += 1
                elif p in CONSONANT_IDS:
                    consonant_err += 1
            # deletions (extra true phones)
            for p in true_seg[min_len:]:
                per_phone_err[p] += 1
                if p in VOWEL_IDS:
                    vowel_err += 1
                elif p in CONSONANT_IDS:
                    consonant_err += 1
            # insertions (extra predicted phones)
            for p in pred_seg[min_len:]:
                per_phone_err[p] += 1
                if p in VOWEL_IDS:
                    vowel_err += 1
                elif p in CONSONANT_IDS:
                    consonant_err += 1
        elif tag == "delete":
            for p in true_seg:
                per_phone_err[p] += 1
                if p in VOWEL_IDS:
                    vowel_err += 1
                elif p in CONSONANT_IDS:
                    consonant_err += 1
        elif tag == "insert":
            for p in pred_seg:
                per_phone_err[p] += 1
                if p in VOWEL_IDS:
                    vowel_err += 1
                elif p in CONSONANT_IDS:
                    consonant_err += 1

    return vowel_err, vowel_count, consonant_err, consonant_count

def dann_alpha(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None) -> float:
    """
    Standard DANN schedule: alpha goes from ~0 -> 1 over training.
    """
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

def dann_alpha_alt(step: int, total_steps: int, gamma: float = 10.0, alpha_max_steps: int = None, phase: int = 16) -> float:
    """
    Alternative DANN schedule: alpha goes from ~0 -> 1 over training as a faster oscillating sinusoidal function.
    This version oscillates twice as fast compared to the original (frequency doubled).
    """
    if alpha_max_steps is None:
        alpha_max_steps = total_steps
    p = min(float(step) / float(alpha_max_steps), 1.0)
    # Double the frequency: 4*pi instead of 2*pi
    return float(0.5 + 0.5 * math.sin(phase * math.pi * p - math.pi/2))


def move_to_device(x: Any, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def unpack_batch_5or7(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Your loader sometimes returns 5 items, sometimes 7:
      5: X, y, X_len, y_len, dayIdx
      7: X, y, X_len, y_len, dayIdx, y2, y2_len
    If 7 items are present, use y2/y2_len as the labels for training/eval.
    """
    if len(batch) == 7:
        X, y, X_len, y_len, dayIdx, y2, y2_len = batch
        # Use phoneme labels when present (because y is character)
        y, y_len = y2, y2_len
    else:
        X, y, X_len, y_len, dayIdx = batch
        y2, y2_len = None, None
    return X, X_len, y, y_len, dayIdx, y2, y2_len

def build_phone_prior(train_targets, num_classes, blank_id=0, device="cpu"):
    """
    Args:
        train_targets: 1D tensor of all target ids concatenated
        num_classes: C (including blank)
    Returns:
        p_prior: (C-1,) tensor excluding blank, normalized
    """
    counts = torch.bincount(train_targets, minlength=num_classes).float()

    # remove blank
    counts[blank_id] = 0.0

    p_prior = counts[1:]  # exclude blank
    p_prior = p_prior / (p_prior.sum() + 1e-8)

    return p_prior.to(device)

 
def trainModelDANN(args, model):
    
    # Check if distributed training is enabled (before wandb init)
    is_distributed = dist.is_initialized() if dist.is_available() else False
    rank = dist.get_rank() if is_distributed else 0
    
    # CRITICAL: Disable WandB on all non-rank-0 processes to prevent auto-initialization
    if is_distributed and rank != 0:
        os.environ['WANDB_MODE'] = 'disabled'
        print(f"ℹ️  Rank {rank}: WandB disabled (only rank 0 logs)")
    else:
        # FORCE online mode - never allow offline (only on rank 0)
        os.environ['WANDB_MODE'] = 'online'
        # Remove any offline settings that might exist
        if 'WANDB_DIR' in os.environ:
            # Don't change WANDB_DIR, but ensure mode is online
            pass
    
    # Get wandb project and entity from args, with fallback defaults
    wandb_project = args.get('wandb_project', "Neural Decoder")
    wandb_entity = args.get('wandb_entity', "skaasyap-ucla")
    wandb_name = args.get('wandb_name', args['modelName'])
    
    # Only initialize wandb on rank 0
    if is_distributed and rank != 0:
        wandb_initialized = False
        # Ensure wandb is disabled on this rank
        if wandb.run is not None:
            wandb.finish()
    else:
        # Print what we're logging to
        print(f"\n📊 Logging to WandB (ONLINE ONLY):")
        print(f"   Project: {wandb_project}")
        print(f"   Entity: {wandb_entity}")
        print(f"   Run Name: {wandb_name}\n")
        
        # Try to login first if API key is available
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            try:
                wandb.login(key=wandb_api_key, relogin=True)
                print("✅ WandB login successful")
            except Exception as e:
                print(f"⚠️  WandB login warning: {e}")
        else:
            print("⚠️  WANDB_API_KEY not found in environment")
    
    # Prepare wandb init kwargs - ALWAYS online, never offline
    wandb_init_kwargs = {
        'project': wandb_project,
        'config': dict(args),
        'name': wandb_name,
        'mode': 'online'  # ALWAYS online - never offline
    }
    
    # Only add entity if it's not None (wandb uses default entity from API key if not specified)
    if wandb_entity is not None:
        wandb_init_kwargs['entity'] = wandb_entity
    
    if len(args['wandb_id']) > 0 and args['start_epoch'] != 0:
        wandb_init_kwargs['resume'] = "must"
        wandb_init_kwargs['id'] = args["wandb_id"]
    
    # Initialize WandB in ONLINE mode only (only on rank 0)
    # CRITICAL: Only initialize on rank 0, all other ranks should have WANDB_MODE=disabled
    wandb_mode = os.environ.get('WANDB_MODE', 'online').lower()
    if is_distributed and rank != 0:
        wandb_initialized = False
        # Double-check that wandb is disabled and not initialized
        if wandb.run is not None:
            wandb.finish()
    elif wandb_mode == 'disabled':
        # Extra safety: if WANDB_MODE is disabled, don't initialize even on rank 0
        wandb_initialized = False
        print("⚠️  WandB is disabled via WANDB_MODE environment variable")
    else:
        # Only rank 0 reaches here - initialize WandB
        wandb_initialized = False
        try:
            wandb.init(**wandb_init_kwargs)
            # Verify it's actually online
            if wandb.run is not None and wandb.run.url is not None:
                wandb_initialized = True
                print("✅ WandB initialized in ONLINE mode")
            else:
                raise Exception("WandB run created but no URL (might be offline)")
        except wandb.errors.CommError as e:
            # Permission error during bucket creation - but run might still be created
            print(f"⚠️  WandB permission error (bucket creation): {e}")
            time.sleep(2)  # Give wandb time to create the run
            
            # Check if run was created despite the error
            if wandb.run is not None and wandb.run.url is not None:
                wandb_initialized = True
                print("   ✅ Run was created despite error - continuing in ONLINE mode")
            else:
                # The run wasn't created - this might be a permission issue with the project
                print("   ❌ Run not created or not online. This may be a permission issue.")
                print("   ❌ WandB ONLINE mode is required - cannot proceed without online logging.")
                raise RuntimeError("WandB failed to initialize in ONLINE mode. Please check permissions.")
        except Exception as e:
            print(f"❌ WandB initialization error: {e}")
            time.sleep(1)
            if wandb.run is None or wandb.run.url is None:
                print("   ❌ WandB ONLINE mode is required - cannot proceed without online logging.")
                raise RuntimeError("WandB failed to initialize in ONLINE mode.")
            else:
                wandb_initialized = True
                print("   ✅ WandB run created successfully")
        
        # Print WandB connection details - verify it's online (only on rank 0)
        if wandb.run is None or wandb.run.url is None:
            print("❌ WandB run not properly initialized in ONLINE mode")
            raise RuntimeError("WandB must be initialized in ONLINE mode to continue")
        
        # Verify it's online (has URL means it's online)
        if os.environ.get('WANDB_MODE', 'online').lower() != 'online':
            print("❌ WandB mode is not set to 'online'")
            raise RuntimeError("WandB must be in ONLINE mode")
        
        print("\n" + "="*80)
        print("🔗 WandB Connection Details (ONLINE):")
        # Safely get username from wandb viewer
        try:
            viewer = wandb.api.viewer() if hasattr(wandb.api, 'viewer') else None
            username = viewer.get('username', 'victoriazhang') if viewer and isinstance(viewer, dict) else 'victoriazhang'
        except (KeyError, AttributeError, Exception):
            username = 'victoriazhang'
        print(f"   ✅ Logged in as: {username}")
        print(f"   Project: {wandb.run.project}")
        print(f"   Entity: {wandb.run.entity}")
        print(f"   Run Name: {wandb.run.name}")
        print(f"   Run ID: {wandb.run.id}")
        print(f"   Run URL: {wandb.run.url}")
        print(f"   Mode: ONLINE (enforced)")
        print("="*80 + "\n")
        print("✅ WandB is running ONLINE and will sync to your dashboard in real-time!")

    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # Check if distributed training is enabled
    is_distributed = dist.is_initialized() if dist.is_available() else False
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # DANN: Create source and target loaders
    # For DANN, we need source (labeled) and target (unlabeled) domains
    # Determine source and target days (same for all ranks)
    if "ptDecoder_ctc_both_held_out_days" in args["datasetPath"]:
        source_days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        target_days = [15, 16, 17, 18, 19]
    if "ptDecoder_ctc_both_held_out_days_big_0" in args["datasetPath"]:
        source_days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        target_days = [12, 13, 14, 15, 16, 17, 18, 19]
    if "ptDecoder_ctc_both_held_out_days_big_1" in args["datasetPath"]:
        source_days = [0, 1, 2, 3, 4, 5, 6, 7]
        target_days = [12, 13, 14, 15, 16, 17, 18, 19]
    elif "ptDecoder_ctc_both_held_out_days_interleaved" in args["datasetPath"]:
        source_days = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
        target_days = [0, 2, 8, 14, 19]
    elif "ptDecoder_ctc_both_held_out_days_big_2" in args["datasetPath"]:
        source_days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        target_days = [16, 17, 18, 19]
    # Add new split logic for custom DATA_PATH_KEY splits from train_align.py,
    # i.e. DATA_PATH_KEY == "...big_15_1_4", etc.
    # Note: The pattern is to check if the datasetPath contains the relevant unique string.
    if "ptDecoder_ctc_both_held_out_days_big_15_1_4" in args["datasetPath"]:
        source_days = list(range(0, 15))
        target_days = list(range(16, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_16_0_4" in args["datasetPath"]:
        source_days = list(range(0, 16))
        target_days = list(range(16, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_16_1_3" in args["datasetPath"]:
        source_days = list(range(0, 16))
        target_days = list(range(17, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_17_0_3" in args["datasetPath"]:
        source_days = list(range(0, 17))
        target_days = list(range(17, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_17_1_2" in args["datasetPath"]:
        source_days = list(range(0, 17))
        target_days = list(range(18, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_18_0_2" in args["datasetPath"]:
        source_days = list(range(0, 18))
        target_days = list(range(18, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_19_0_1" in args["datasetPath"]:
        source_days = list(range(0, 19))
        target_days = list(range(19, 20))
    elif "ptDecoder_ctc_both_held_out_days_big_18_1_1" in args["datasetPath"]:
        source_days = list(range(0, 18))
        target_days = list(range(19, 20))
    elif "ptDecoder_ctc_both_held_out_days_12_0_4" in args["datasetPath"]:
        source_days = list(range(0, 12))
        target_days = list(range(12, 16))
    elif "ptDecoder_ctc_both_held_out_days_12_0_1" in args["datasetPath"]:
        source_days = list(range(0, 12))
        target_days = list(range(12, 13))
    elif "ptDecoder_ctc_both_held_out_days_16_0_1" in args["datasetPath"]:
        source_days = list(range(0, 16))
        target_days = list(range(16, 17))
    elif "ptDecoder_ctc_both_held_out_days_12_8_3_no_test" in args["datasetPath"]:
        source_days = list(range(0, 12))
        target_days = list(range(20, 23))
    elif "ptDecoder_ctc_both_held_out_days_12_2gap_4_2gap_3_test" in args["datasetPath"]:
        source_days = list(range(0, 12))
        target_days = list(range(14, 18))
    else:
        source_days = args.get('source_days')
        target_days = args.get('target_days')
        if source_days is None or target_days is None:
            raise ValueError(f"Unsupported dataset path: {args['datasetPath']} and source_days/target_days not provided")
    
    if rank == 0:
        print(f"📦 Creating DANN data loaders...")
        print(f"   Source days: {source_days}")
        print(f"   Target days: {target_days}")
    
    # Get loaders - source uses trainLoader, target uses testLoader
    trainLoader, testLoader, loadedData = getDatasetLoaders(
            args["datasetPath"],
            args["batchSize"],
            args['restricted_days'], 
            args['ventral_6v_only'],
            include_original=args.get('include_original', True),
            include_stretched_samples=args.get('include_stretched_samples', False),
            include_prolonged_samples=args.get('include_prolonged_samples', False),
            stretch_range=args.get('stretch_range', 2.0),
            distributed=False,
    )
    source_trainLoader = trainLoader

    # Use competition hold-out split as the target domain if available
    competition_hold_data = loadedData.get("competition")
    if competition_hold_data is not None:
        base_target_ds = SpeechDataset(
            competition_hold_data,
            restricted_days=args['restricted_days'],
            ventral_6v_only=args['ventral_6v_only']
        )
        include_stretched = args.get('include_stretched_samples', False)
        include_original = args.get('include_original', True)
        stretch_range = args.get('stretch_range', 2.0)
        if include_stretched:
            stretched_target_ds = StretchSqueezeDataset(
                base_target_ds,
                stretch_range=stretch_range,
            )
            if include_original:
                target_ds = ConcatDataset([base_target_ds, stretched_target_ds])
            else:
                target_ds = stretched_target_ds
        else:
            target_ds = base_target_ds
        target_trainLoader = torch.utils.data.DataLoader(
            target_ds,
            batch_size=args["batchSize"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=testLoader.collate_fn,
        )
        if rank == 0:
            print("✅ Using competition hold-out split as target_trainLoader")
    else:
        target_trainLoader = testLoader
        if rank == 0:
            print("ℹ️  Competition hold-out split not found; using test split as target_trainLoader")

    rep_layer_idx = args.get('rep_layer_idx', args['depth']-1)
    
    # Create mapping from session names to unique domain IDs
    # Only use source (training) sessions for discriminator training
    source_session_names = [sessionNames[i] for i in sorted(source_days)]
    
    # Create mapping: session_name -> unique_domain_id (0 to n_source_sessions-1)
    # Only source sessions get domain IDs for discriminator training
    session_to_domain_id = {session_name: idx for idx, session_name in enumerate(source_session_names)}
    
    # Create mappings from relative day indices (in train/test splits) to actual session names
    # For train (source): day_idx 0 -> source_days[0] -> sessionNames[source_days[0]]
    # For test (target): day_idx 0 -> target_days[0] -> sessionNames[target_days[0]]
    train_day_to_session = {rel_day_idx: sessionNames[source_days[rel_day_idx]] 
                           for rel_day_idx in range(len(loadedData["train"]))}
    test_day_to_session = {rel_day_idx: sessionNames[target_days[rel_day_idx]] 
                          for rel_day_idx in range(len(loadedData["test"]))}
    
    if rank == 0:
        print(f"📋 Session to Domain ID mapping (source/training sessions only):")
        for session_name, domain_id in session_to_domain_id.items():
            print(f"   {session_name} -> {domain_id}")
        print(f"   Total source domains: {len(session_to_domain_id)}")
        print(f"   Target sessions (unseen): {[sessionNames[i] for i in sorted(target_days)]}")
    
    # Create a separate loader for train metrics (full dataset, no sampler) to mimic single GPU
    # This ensures train CER is computed on the same data as single GPU mode
    trainMetricsLoader = source_trainLoader  # Full loader for metrics computation only
    
    # Wrap train loaders with DistributedSampler if using distributed training
    train_sampler = None
    source_train_sampler = None
    target_train_sampler = None
    if is_distributed:

        batch_size_per_gpu = args["batchSize"] // world_size
        if args["batchSize"] % world_size != 0:
            print(f"⚠️  Warning: batch_size {args['batchSize']} is not divisible by {world_size} GPUs")
            print(f"   Using {batch_size_per_gpu} samples per GPU (total effective batch size: {batch_size_per_gpu * world_size})")
        
        source_train_sampler = DistributedSampler(
            source_trainLoader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        source_trainLoader = torch.utils.data.DataLoader(
            source_trainLoader.dataset,
            batch_size=batch_size_per_gpu,  # Per GPU batch size
            sampler=source_train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=source_trainLoader.collate_fn,
        )
        
        target_train_sampler = DistributedSampler(
            target_trainLoader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        target_trainLoader = torch.utils.data.DataLoader(
            target_trainLoader.dataset,
            batch_size=batch_size_per_gpu,  # Per GPU batch size
            sampler=target_train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=target_trainLoader.collate_fn,
        )
        
        print(f"✅ Using DistributedSampler for training data (rank {rank}/{world_size-1})")
        print(f"   Batch size per GPU: {batch_size_per_gpu} (total effective: {batch_size_per_gpu * world_size})")
        print(f"   Train metrics will be computed on full dataset (rank 0 only) to match single GPU behavior")
    
    # Create a separate evaluation loader (no DistributedSampler, full dataset, used only by rank 0)
    eval_batch_size = args.get('eval_batch_size', args["batchSize"])

    # Evaluation loader: always use the original test split
    eval_testLoader = torch.utils.data.DataLoader(
        testLoader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # No shuffle for evaluation
        num_workers=0,
        pin_memory=True,
        collate_fn=testLoader.collate_fn,
    )
    if rank == 0:
        print("✅ Evaluating on test split (eval_testLoader)")

    # Create fixed probe subsets (64 source + 64 eval by default) for tracking representation drift
    probe_num = int(args.get("probe_samples_per_domain", 64))
    probe_seed = int(args.get("probe_seed", args.get("seed", 42)))
    probe_generator = torch.Generator()
    probe_generator.manual_seed(probe_seed)
    probe_src_indices = torch.randperm(len(source_trainLoader.dataset), generator=probe_generator)[
        : min(probe_num, len(source_trainLoader.dataset))
    ].tolist()
    probe_eval_indices = torch.randperm(len(eval_testLoader.dataset), generator=probe_generator)[
        : min(probe_num, len(eval_testLoader.dataset))
    ].tolist()

    probe_src_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(source_trainLoader.dataset, probe_src_indices),
        batch_size=min(probe_num, args["batchSize"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=source_trainLoader.collate_fn,
    )
    probe_eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(eval_testLoader.dataset, probe_eval_indices),
        batch_size=min(probe_num, eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=eval_testLoader.collate_fn,
    )

    if rank == 0:
        print(
            f"📊 Fixed probe sets created (source idxs head: {probe_src_indices[:5]}, eval idxs head: {probe_eval_indices[:5]})"
        )
    
    
    # Watch the model (only if wandb is initialized and on rank 0)
    if wandb.run is not None and (not is_distributed or rank == 0):
        wandb.watch(model.module if is_distributed and hasattr(model, 'module') else model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # -----------------------
    # DANN-SPECIFIC CONFIG
    # -----------------------
    device = torch.device(args.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    model.to(device)

    def _flatten_grads(params, grads):
        """
        Flatten gradients for a list of parameters; replaces missing grads with zeros
        so dot products stay aligned across params.
        """
        flat = []
        for p, g in zip(params, grads):
            if g is None:
                flat.append(torch.zeros_like(p, device=p.device).reshape(-1))
            else:
                flat.append(g.detach().reshape(-1))
        if len(flat) == 0:
            return torch.tensor([], device=device)
        return torch.cat(flat)

    def _compute_probe_mean_and_cov(probe_loader):
        feats = []
        with torch.no_grad():
            for batch in probe_loader:
                batch = move_to_device(batch, device)
                X, X_len, y, y_len, dayIdx, y2, y2_len = unpack_batch_5or7(batch)
                # Noise augmentation is faster on GPU
                if args.get('noise_augmentation') == True:
                    if args["whiteNoiseSD"] > 0:
                        X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

                    if args["constantOffsetSD"] > 0:
                        X += (
                            torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                            * args["constantOffsetSD"]
                        )
                adjustedLens = base_model.compute_length(X_len)
                _, rep, _ = base_model(X, X_len, dayIdx, return_rep=True, rep_layer_idx=rep_layer_idx)
                pooled = masked_mean_pool(rep, adjustedLens)
                feats.append(pooled.detach().cpu())
        if len(feats) == 0:
            return None, None
        feats = torch.cat(feats, dim=0)
        mean = feats.mean(dim=0)
        if feats.shape[0] < 2:
            cov = torch.zeros((feats.shape[1], feats.shape[1]), device=feats.device)
        else:
            cov = torch.cov(feats.T)
        return mean, cov
    
    lambda_domain = args.get('dann_lambda', 0.001)
    lambda_src = args.get('lambda_src', 1.0)
    lambda_tgt = args.get('lambda_tgt', 1.0)
    dann_warmup_epochs = args.get('dann_warmup_epochs', 0)  # Warmup epochs with lambda=0
    domain_hidden = args.get('dann_hidden', 256)
    domain_dropout = args.get('domain_dropout', 0.1)
    use_amp = args.get('use_amp', True)
    grad_clip = args.get('grad_clip', None)
    # Discriminator LR multiplier (2x to 10x encoder LR)
    dann_lr_multiplier = args.get('dann_lr_multiplier', 1.0)
    dann_weight_decay = args.get('dann_weight_decay', None)  # If None, use same as encoder
    use_target_loss = args.get('use_target_loss', True)  # Whether to use target domain loss (entropy maximization)
    linear_discriminator = args.get('linear_discriminator', True)  # If True, use linear discriminator; if False, use MLP
    dann_dropout_channel_prob = args.get('dann_dropout_channel_prob', 0.0)  # Probability of randomly masking timesteps before pooling (0.0 = disabled)
    bottleneck_dim = args.get('bottleneck_dim', None)
    truely_mdan = bool(args.get('truely_mdan', False))
    use_spectral_norm = bool(args.get('use_spectral_norm', False))
    phase = args.get('phase', 16)
    weighted_ctc_loss = bool(args.get('weighted_ctc_loss', False))
    kl_phone_prior_loss_weight = args.get('kl_phone_prior_loss_weight', 0.0)
    nClasses = args.get('nClasses', 40) + 1
    reduce_entropy_loss_weight = args.get('reduce_entropy_loss_weight', 0.0)
    
    # Train discriminator only on training (source) sessions
    n_domains = len(source_days)
    
    if rank == 0:
        print(f"args: {args}")
        print(f"🔧 DANN Training Configuration:")
        print(f"   Lambda domain: {lambda_domain}")
        print(f"   Use target loss: {use_target_loss}")
        if use_target_loss:
            print(f"   Lambda target: {lambda_tgt}")
        print(f"   Warmup epochs: {dann_warmup_epochs} (lambda=0 during warmup)")
        print(f"   Discriminator LR multiplier: {dann_lr_multiplier}x")
        if dann_weight_decay is not None:
            print(f"   Discriminator weight decay: {dann_weight_decay} (encoder: {args['l2_decay']})")
        else:
            print(f"   Discriminator weight decay: {args['l2_decay']} (same as encoder)")
        print(f"   Discriminator type: {'Linear' if linear_discriminator else 'MLP'} (hidden_dim={domain_hidden}, dropout={domain_dropout})")
        if use_spectral_norm:
            print(f"   Spectral norm: enabled on discriminator linear layers")
        if dann_dropout_channel_prob > 0.0:
            print(f"   Timestep dropout: {dann_dropout_channel_prob} (randomly mask timesteps before pooling)")
        if truely_mdan:
            print(f"   MDAN mode: K binary classifiers (source-i vs target), no k+1 classifier")
        else:
            print(f"   MDAN mode: single multiclass classifier (source-i vs other sources + target)")
    
    # Create domain discriminator (only for source domains)
    if bottleneck_dim is not None:
        in_dim = bottleneck_dim
    else:
        in_dim = args['dim']
    def apply_spectral_norm_to_linears(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.utils.spectral_norm(m)

    if truely_mdan:
        # MDAN: K binary discriminators (each source-i vs target)
        domain_disc = torch.nn.ModuleList([
            DomainDiscriminator(
                in_dim=in_dim,
                hidden_dim=domain_hidden,
                n_domains=1,
                dropout=domain_dropout,
                linear_discriminator=linear_discriminator,
            ) for _ in range(n_domains)
        ]).to(device)
        if use_spectral_norm:
            for disc in domain_disc:
                apply_spectral_norm_to_linears(disc)
       
    else:
        domain_disc = DomainDiscriminator(
            in_dim=in_dim,
            hidden_dim=domain_hidden,
            n_domains=n_domains + 1,
            dropout=domain_dropout,
            linear_discriminator=linear_discriminator,
        ).to(device)
        if use_spectral_norm:
            apply_spectral_norm_to_linears(domain_disc)

    # -----------------------
    # FUTURE PREDICTION CONFIG (CPC-style)
    # -----------------------
    use_future_pred = bool(args.get("use_future_pred_loss", False))
    future_pred_weight = float(args.get("future_pred_weight", 0.0))
    future_pred_steps = args.get("future_pred_steps", [1, 2])

    # Get underlying model (for DDP compatibility)
    base_model = model.module if is_distributed and hasattr(model, "module") else model
    future_predictor = getattr(base_model, "future_predictor", None)

    if use_future_pred and future_pred_weight > 0.0 and future_predictor is not None:
        print(f"✅ Using future prediction loss with steps={future_pred_steps}, "
              f"weight={future_pred_weight}")
    elif use_future_pred and future_pred_weight > 0.0 and future_predictor is None:
        print("⚠️ use_future_pred_loss=True but model.future_predictor is None. "
              "Did you construct the model with use_future_pred_loss=True?")
    
    # -----------------------
    # SAME-PHONEME CONTRASTIVE CONFIG
    # -----------------------
    use_phone_contrastive = bool(args.get("use_phone_contrastive_loss", False))
    phone_contrastive_weight = float(args.get("phone_contrastive_weight", 0.0))
    phone_contrastive_max_samples = int(args.get("phone_contrastive_max_samples", 512))
    phone_contrastive_temperature = float(args.get("phone_contrastive_temperature", 0.1))
    phone_contrastive_start_epoch = int(args.get("phone_contrastive_start_epoch", 0))

    if use_phone_contrastive and phone_contrastive_weight > 0.0:
        use_cross_trial_phone_contrastive_loss = bool(args.get("use_cross_trial_phone_contrastive_loss", True))
        phone_contrastive_conf_thresh = float(args.get("phone_contrastive_conf_thresh", 0.7))
        print(
            f"✅ Using phone contrastive loss: "
            f"λ={phone_contrastive_weight}, max_samples={phone_contrastive_max_samples}, "
            f"T={phone_contrastive_temperature}, start_epoch={phone_contrastive_start_epoch}, "
            f"use_cross_trial_phone_contrastive_loss={use_cross_trial_phone_contrastive_loss}, "
            f"phone_contrastive_conf_thresh={phone_contrastive_conf_thresh}"
        )
    use_cross_trial_phone_contrastive_loss = bool(args.get("use_cross_trial_phone_contrastive_loss", True))
    phone_contrastive_conf_thresh = float(args.get("phone_contrastive_conf_thresh", 0.7))
    detach_discriminator_from_encoder = bool(args.get("detach_discriminator_from_encoder", False))

    # -----------------------
    # NEXT-TOKEN PREDICTION (NTP) CONFIG
    # -----------------------
    use_ntp = bool(args.get("use_ntp", False))
    lambda_ntp = float(args.get("lambda_ntp", 0.0))
    
    # Detect if model supports NTP
    base_model = model.module if is_distributed and hasattr(model, "module") else model
    model_has_ntp = (
        getattr(base_model, "use_ntp", False) or
        hasattr(base_model, "ntp_head") or
        base_model.__class__.__name__ == "DeepCTCNTPLocalGlobalViT_Phoneme"
    )
    
    if use_ntp and lambda_ntp > 0.0 and model_has_ntp:
        print(f"✅ Using NTP loss: λ={lambda_ntp}")
    elif use_ntp and lambda_ntp > 0.0 and not model_has_ntp:
        print("⚠️ use_ntp=True and lambda_ntp>0 but model does not support NTP. "
              "Did you construct the model with use_ntp=True?")

    # DANN: Separate optimizers for model and discriminator
    # Encoder/CTC head optimizer
    model_params = list(model.parameters())
    disc_params = list(domain_disc.parameters())
    
    if args['AdamW']:
        print("USING ADAMW")
        # Encoder optimizer
        model_optimizer = torch.optim.AdamW(
            model_params, 
            lr=float(args['lrStart']), 
            weight_decay=float(args['l2_decay']), 
            betas=(float(args['beta1']), float(args['beta2'])),
        )
        # Discriminator optimizer with higher LR
        disc_lr = float(args['lrStart']) * dann_lr_multiplier
        disc_wd = dann_weight_decay if dann_weight_decay is not None else float(args['l2_decay'])
        disc_optimizer = torch.optim.AdamW(
            disc_params,
            lr=disc_lr,
            weight_decay=disc_wd,
            betas=(0.9, 0.999),
        )
        if rank == 0:
            print(f"   Encoder LR: {args['lrStart']}, Discriminator LR: {disc_lr} ({dann_lr_multiplier}x)")
    else:
        if args['SOAP']:
            print("USING SOAP")
            from .soap import SOAP
            model_optimizer = SOAP(
                model_params,
                lr=float(args['lrStart']),
                betas=(0.95, 0.95),
                weight_decay=float(args['l2_decay']),
                precondition_frequency=int(args.get('precondition_frequency', 10)),
            )
            disc_lr = float(args['lrStart']) * dann_lr_multiplier
            disc_wd = dann_weight_decay if dann_weight_decay is not None else float(args['l2_decay'])
            disc_optimizer = torch.optim.AdamW(  # Use AdamW for discriminator even if model uses SOAP
                disc_params,
                lr=disc_lr,
                weight_decay=disc_wd,
                betas=(float(args['beta1']), float(args['beta2'])),
            )
            if rank == 0:
                print(f"   Encoder LR: {args['lrStart']}, Discriminator LR: {disc_lr} ({dann_lr_multiplier}x)")
        else:
            print("USING VANILLA ADAM")
            model_optimizer = torch.optim.Adam(
                model_params,
                lr=float(args["lrStart"]),
                betas=(0.9, 0.999),
                eps=0.1,
                weight_decay=float(args["l2_decay"]),
            )
            disc_lr = float(args['lrStart']) * dann_lr_multiplier
            disc_wd = dann_weight_decay if dann_weight_decay is not None else float(args["l2_decay"])
            disc_optimizer = torch.optim.Adam(
                disc_params,
                lr=disc_lr,
                betas=(0.9, 0.999),
                eps=0.1,
                weight_decay=disc_wd,
            )
            if rank == 0:
                print(f"   Encoder LR: {args['lrStart']}, Discriminator LR: {disc_lr} ({dann_lr_multiplier}x)")
    
    # For backward compatibility, create a combined optimizer view
    # (but we'll use separate optimizers in training loop)
    optimizer = model_optimizer  # Keep for scheduler compatibility
    
    if args['learning_scheduler'] == 'multistep': 
        print("Multistep scheduler")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=float(args['gamma']),
        )
        
    elif args['learning_scheduler'] == 'cosine':
        print("Cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args['n_epochs']),     # Total epochs to decay over
            eta_min=float(args['eta_min'])    # Final learning rate
        )
            
    elif args['learning_scheduler'] == 'warmcosine':
        print("Warm Cosine Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(args['T_0']),       # first cosine decay cycle
            T_mult=int(args['T_mult']), # multiplier
            eta_min=float(args['eta_min'])
        )
        
    else:
        print("Linear scheduler")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrStart"] / args["lrStart"],
            total_iters=args["n_epochs"],
        )
    
    if len(args['load_pretrained_model']) > 0:
        optimizer_path = os.path.join(args['load_pretrained_model'], 'optimizer')
        # Try to load model optimizer
        try:
            model_optimizer.load_state_dict(torch.load(optimizer_path, map_location=args['device']))
            if rank == 0:
                print(f"Loaded model optimizer state from {args['load_pretrained_model']}")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load model optimizer state: {e}")
        
        # Try to load discriminator optimizer (optional, will start fresh if not found)
        disc_optimizer_path = os.path.join(args['load_pretrained_model'], 'disc_optimizer')
        if os.path.exists(disc_optimizer_path):
            try:
                disc_optimizer.load_state_dict(torch.load(disc_optimizer_path, map_location=args['device']))
                if rank == 0:
                    print(f"Loaded discriminator optimizer state from {args['load_pretrained_model']}")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load discriminator optimizer state: {e}")
        else:
            if rank == 0:
                print(f"Discriminator optimizer not found, starting fresh")
        
        scheduler_path = os.path.join(args['load_pretrained_model'], 'scheduler')
        try:
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=args['device']))
            if rank == 0:
                print(f"Loaded scheduler state from {args['load_pretrained_model']}")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load scheduler state: {e}")
        
    # --train--
    testLoss = []
    testCER = []
    
    # Get base model for DDP compatibility
    base_model = model.module if is_distributed and hasattr(model, "module") else model
    encoder_params = [p for p in model.parameters() if p.requires_grad]
    grad_metrics_eps = 1e-8

    # Pre-compute phone prior once from the full training set
    p_prior = None
    if kl_phone_prior_loss_weight > 0:
        if rank == 0:
            print("Computing phone prior from full training set...")
            train_targets = []
            with torch.no_grad():
                for batch in trainMetricsLoader:
                    X, X_len, y, y_len, dayIdx, y2, y2_len = unpack_batch_5or7(batch)
                    if y is None:
                        continue
                    for i in range(y.shape[0]):
                        length = int(y_len[i].item()) if torch.is_tensor(y_len) else int(y_len[i])
                        if length > 0:
                            train_targets.append(y[i, :length].reshape(-1).cpu())
            if len(train_targets) == 0:
                raise ValueError("No training targets found to compute phone prior.")
            p_prior = build_phone_prior(
                torch.cat(train_targets, dim=0),
                nClasses,
                blank_id=BLANK_TOKEN,
                device=device,
            )
        if is_distributed:
            if p_prior is None:
                p_prior = torch.zeros(nClasses - 1, device=device)
            dist.broadcast(p_prior, src=0)
    
    # DANN: Training loop setup
    steps_per_epoch = len(source_trainLoader)
    total_steps = steps_per_epoch * (args['n_epochs'] - args["start_epoch"])
    alpha_max_steps = None if args.get('alpha_max_epochs', None) is None else int(args.get('alpha_max_epochs') * steps_per_epoch)
    global_step = 0

    for epoch in range(args["start_epoch"], args['n_epochs']):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed:
            if source_train_sampler is not None:
                source_train_sampler.set_epoch(epoch)
            if target_train_sampler is not None:
                target_train_sampler.set_epoch(epoch)

        # Warmup: set lambda_domain to 0 for first dann_warmup_epochs
        is_warmup_epoch = epoch < dann_warmup_epochs
        current_lambda_domain = 0.0 if is_warmup_epoch else lambda_domain
        if rank == 0 and epoch == dann_warmup_epochs and dann_warmup_epochs > 0:
            print(f"🔥 Warmup complete! Starting domain loss at epoch {epoch+1} with lambda={lambda_domain}")

        epoch_start_time = time.time()  # Track time for this epoch
        train_task_losses = []
        train_mdan_losses = []
        train_src_dom_losses = []
        train_tgt_dom_losses = []
        train_dom_accuracies = []
        train_total_losses = []
        train_encoder_grad_norms = []
        train_disc_grad_norms = []
        train_alpha_values = []
        train_tgt_none_rates = []
        train_tgt_max_logits = []
        train_tgt_entropies = []
        train_tgt_frac_boundaries = []
        train_src_logits_mins = []
        train_src_logits_maxs = []
        train_tgt_logits_mins = []
        train_tgt_logits_maxs = []
        train_task_grad_norms = []
        train_dom_grad_norms = []
        train_dom_task_grad_ratios = []
        train_dom_task_grad_coss = []
        train_dom_correct_mdan = 0
        train_dom_total_mdan = 0
        model.train()
        domain_disc.train()
        num_batches = 0
        train_dom_correct = 0
        train_dom_total = 0

        use_mdan_heads = isinstance(domain_disc, torch.nn.ModuleList)
        K = len(domain_disc) if use_mdan_heads else 0
        if use_mdan_heads:
            head_src_correct = torch.zeros(K, device=device)
            head_src_total   = torch.zeros(K, device=device)
            head_tgt_correct = torch.zeros(K, device=device)
            head_tgt_total   = torch.zeros(K, device=device)
        else:
            head_src_correct = None
            head_src_total = None
            head_tgt_correct = None
            head_tgt_total = None


        # Cycle target loader forever
        target_iter = iter(target_trainLoader)

        # Only show progress bar on rank 0 to avoid interference in distributed training
        progress_bar = tqdm(source_trainLoader, desc=f"Epoch {epoch+1}/{args['n_epochs']}", disable=(is_distributed and rank != 0))
        for step, src_batch in enumerate(progress_bar):
            num_batches += 1

            src_batch = move_to_device(src_batch, device)

            Xs, Xs_len, ys, ys_len, dayS, ys2, ys2_len = unpack_batch_5or7(src_batch)
            

            # Noise augmentation is faster on GPU
            if args.get('noise_augmentation') == True:
                if args["whiteNoiseSD"] > 0:
                    Xs += torch.randn(Xs.shape, device=args["device"]) * args["whiteNoiseSD"]

                if args["constantOffsetSD"] > 0:
                    Xs += (
                        torch.randn([Xs.shape[0], 1, Xs.shape[2]], device=args["device"])
                        * args["constantOffsetSD"]
                    )

            # Compute adjusted lengths for source and target
            adjustedLens_src = base_model.compute_length(Xs_len)

            # Compute alpha value for GRL (tracked for all iterations, even during warmup)
            warmup_steps = dann_warmup_epochs * steps_per_epoch
            if dann_warmup_epochs > 0:
                alpha_total_step = max(total_steps - warmup_steps, 1)
            else:
                alpha_total_step = max(total_steps, 1)
            # Start counting alpha schedule only after warmup so it ramps 0→1
            alpha_step = max(global_step - warmup_steps, 0)
            is_warmup_step = global_step < warmup_steps
            if is_warmup_step:
                alpha = 0.0
            else:
                if args.get('dann_alpha_type', 'standard') == 'standard':
                    alpha = dann_alpha(alpha_step, alpha_total_step, alpha_max_steps=alpha_max_steps)
                elif args.get('dann_alpha_type', 'alternative') == 'alternative':
                    alpha = dann_alpha_alt(alpha_step, alpha_total_step, alpha_max_steps=alpha_max_steps)
                else:
                    raise ValueError(f"Invalid dann_alpha_type: {args.get('dann_alpha_type', 'standard')}")

            
            train_alpha_values.append(float(alpha))

            # Zero gradients for encoder optimizers
            model_optimizer.zero_grad(set_to_none=True)
            disc_optimizer.zero_grad(set_to_none=True)


            # ----- SOURCE forward (task + domain) -----
            src_logits, src_rep, src_final = model(Xs, Xs_len, dayS, return_rep=True, rep_layer_idx=rep_layer_idx)

            # Log projector and discriminator weights to wandb at each dimension
            # Projector: model.projection (usually [output_dim, input_dim])
            projector_weights = model.projection.weight.detach().cpu().numpy() if hasattr(model, "projection") else None
            disc_weights = domain_disc.weight.detach().cpu().numpy() if hasattr(domain_disc, "weight") else None

            # Log each dimension (column) as a separate histogram (for projector)
            if projector_weights is not None and step % 10 == 0 and (not is_distributed or rank == 0):  # reduce wandb logging frequency for performance
                for i in range(projector_weights.shape[1]):
                    wandb.log({f"projector/weights_dim_{i}": wandb.Histogram(projector_weights[:, i])}, step=global_step)
            # Log discriminator weights (dimension: [n_domains, in_dim])
            if disc_weights is not None and step % 10 == 0 and (not is_distributed or rank == 0):
                for i in range(disc_weights.shape[1]):
                    wandb.log({f"discriminator/weights_dim_{i}": wandb.Histogram(disc_weights[:, i])}, step=global_step)
            if weighted_ctc_loss:
                class_weights = torch.ones(nClasses)

                vowel_ids = [
                    phoneme_to_id[p] for p in
                    ["OY","AW","AY","EY","OW","ER","UW","AO"]
                ]
                collapse_ids = [phoneme_to_id[p] for p in ["JH","CH","ZH","OY"]]
                for i in collapse_ids:
                    class_weights[i] = 3.0

                task_loss = weighted_forward_ctc(src_logits, adjustedLens_src, ys, ys_len, class_weights)
            else:
                task_loss = forward_ctc(src_logits, adjustedLens_src, ys, ys_len)

            regularization_type = args.get('emb_regularization', None)
            def regularization_loss(x, regularization_type):
                if regularization_type == "l2":
                    return torch.mean(torch.sum(x ** 2, dim=-1))
                elif regularization_type == "variance":
                    eps = 1e-4
                    std = torch.sqrt(x.var(dim=(0,1), unbiased=False) + eps)
                    return torch.mean(F.relu(1.0 - std))
                elif regularization_type == "temporal_smoothness":
                    return torch.mean((x[:, 1:] - x[:, :-1]) ** 2)
                else:
                    raise ValueError(f"Invalid regularization type: {regularization_type}")
            def afn_loss(feat: torch.Tensor, mode: str = "safn", R: float = 25.0, delta_r: float = 1.0):
                """
                feat: [B, D] or [B, T, D] feature vectors (NOT normalized)
                mode: "hafn" or "safn"
                """
                norms = torch.norm(feat, p=2, dim=-1)  # [B, T, C] -> [B, T] or [B, D] -> [B]

                if mode.lower() == "hafn":
                    # (E[||f||] - R)^2
                    return (norms.mean() - R) ** 2

                elif mode.lower() == "safn":
                    # (||f|| - (stopgrad(||f||) + delta_r))^2
                    target = norms.detach() + delta_r
                    return ((norms - target) ** 2).mean()

                else:
                    raise ValueError(f"Unknown AFN mode: {mode}")

            if regularization_type is not None and args.get('emb_regularization_weight') > 0:
                # forward_ctc returns a CPU loss; keep the emb reg term on the same device
                reg_loss = regularization_loss(src_rep, regularization_type).to(task_loss.device)
                task_loss = task_loss + args.get('emb_regularization_weight') * reg_loss
            
            if kl_phone_prior_loss_weight > 0:
                kl_loss = kl_phone_prior_loss(src_logits, p_prior)
                task_loss = task_loss + kl_phone_prior_loss_weight * kl_loss

           

            # Apply random timestep masking if enabled
            if dann_dropout_channel_prob > 0.0:
                src_rep = randomly_mask_channelsteps(src_rep, adjustedLens_src, dann_dropout_channel_prob)
            
            if args.get('non_blank_weighting', False):
                with torch.no_grad():
                    src_logits_for_w = base_model.projection(src_rep) # with or without masking
                    blank_id = 0 
                    p_blank_s = torch.softmax(src_logits_for_w, dim=-1)
                    src_non_blank_w = 1.0 - p_blank_s[..., blank_id]
                
            
            
            if args.get('mean_pool_for_discriminator') is True:
                # Pool [B,T,D] -> [B,D] using adjusted lengths
                # this means discriminator will use a AVERAGED features to discriminate, which may not be useful for task performance
                src_feat = masked_mean_pool(src_rep, adjustedLens_src) # [B, D]
            else:
                src_feat = src_rep # [B, T, D]
            if args.get('normalize_features_for_discriminator', False) and args.get('rep_layer_idx', 0) != 5:
            # Normalize only features sent to discriminator
                src_feat_disc = F.normalize(src_rep, p=2, dim=-1)   # [B,T,D], each frame has ||x||=1
            else:
                src_feat_disc = src_rep
            src_feat_for_discriminator = src_feat_disc.clone().detach()

            # Apply GRL to source features (no GRL during warmup)
            if is_warmup_epoch:
                # During warmup: train discriminator only (no gradients to encoder)
                src_feat_grl = src_feat_disc.detach()
            elif args.get('detach_discriminator_from_encoder') == True:
                # Detach before grad_reverse to stop gradients from discriminator to encoder
                src_feat_grl = src_feat_disc.detach()
            else:
                # Normal DANN: apply grad_reverse to allow reversed gradients back to encoder
                src_feat_grl = grad_reverse(src_feat_disc, alpha=alpha)



            # Prepare target batch/features (reuse for encoder + discriminator steps)
            tgt_feat = None
            tgt_feat_grl = None
            adjustedLens_tgt = None
            if use_target_loss:
                try:
                    tgt_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_trainLoader)
                    tgt_batch = next(target_iter)
                
                tgt_batch = move_to_device(tgt_batch, device)
                Xt, Xt_len, yt, yt_len, dayT, yt2, yt2_len = unpack_batch_5or7(tgt_batch)

                if args.get('noise_augmentation') == True:
                    if args.get('constantOffsetSD') > 0:
                        Xt += (
                            torch.randn([Xt.shape[0], 1, Xt.shape[2]], device=args["device"])
                            * args["constantOffsetSD"]
                        )

                    if args.get('whiteNoiseSD') > 0:
                        Xt += torch.randn(Xt.shape, device=args["device"]) * args["whiteNoiseSD"]
                
                # Compute adjusted lengths for target
                adjustedLens_tgt = base_model.compute_length(Xt_len)
                
                # Forward pass on target (domain only, no task loss)
                if args.get('detach_discriminator_from_encoder') == True:
                    model.eval()
                    with torch.no_grad():
                        tgt_logits, tgt_rep, tgt_final = model(Xt, Xt_len, dayT, return_rep=True, rep_layer_idx=rep_layer_idx)
                    model.train()
                else:
                    tgt_logits, tgt_rep, tgt_final = model(Xt, Xt_len, dayT, return_rep=True, rep_layer_idx=rep_layer_idx)
                
                if reduce_entropy_loss_weight > 0 and tgt_logits is not None and tgt_logits.requires_grad:
                    from .loss import entropy_min_loss
                    # print(f"🎾 Computing entropy minimization loss for target logits")
                    entropy_loss = entropy_min_loss(tgt_logits, adjustedLens_tgt)
                    task_loss = task_loss + reduce_entropy_loss_weight * entropy_loss


              
                # Apply random timestep masking if enabled
                if dann_dropout_channel_prob > 0.0:
                    tgt_rep = randomly_mask_channelsteps(tgt_rep, adjustedLens_tgt, dann_dropout_channel_prob)

                if args.get('non_blank_weighting', False):
                    with torch.no_grad():
                        tgt_logits_for_w = base_model.projection(tgt_rep) # with or without masking
                        blank_id = 0 
                        p_blank_t = torch.softmax(tgt_logits_for_w, dim=-1)
                        tgt_non_blank_w = 1.0 - p_blank_t[..., blank_id]
                
                
                
                if args.get('mean_pool_for_discriminator') is True:
                    # Pool [B,T,D] -> [B,D] using adjusted lengths 
                    # this means discriminator will use a AVERAGED features to discriminate, which may not be useful for task performance
                    tgt_feat = masked_mean_pool(tgt_rep, adjustedLens_tgt)
                else:
                    tgt_feat = tgt_rep
                if args.get('normalize_features_for_discriminator', False) and args.get('rep_layer_idx', 0) != 5:
                    # Normalize only features sent to discriminator
                    tgt_feat_disc = F.normalize(tgt_rep, p=2, dim=-1)   # [B,T,D]
                else:
                    tgt_feat_disc = tgt_rep
                # Apply GRL to target features (no GRL during warmup)
                if is_warmup_epoch:
                    # During warmup: train discriminator only (no gradients to encoder)
                    tgt_feat_grl = tgt_feat_disc.detach()
                elif args.get('detach_discriminator_from_encoder') == True:
                    # Detach before grad_reverse to stop gradients from discriminator to encoder
                    tgt_feat_grl = tgt_feat_disc.detach()
                else:
                    # Normal DANN: apply grad_reverse to allow reversed gradients back to encoder
                    tgt_feat_grl = grad_reverse(tgt_feat_disc, alpha=alpha)

            # Map relative day indices to unique session domain IDs (only for source)
            src_session_names = []
            for day_idx in dayS:
                day_idx_int = int(day_idx.item())
                if day_idx_int not in train_day_to_session:
                    raise ValueError(f"Day index {day_idx_int} not found in train_day_to_session mapping. "
                                   f"Available keys: {list(train_day_to_session.keys())}")
                src_session_names.append(train_day_to_session[day_idx_int])
            
            # Get domain IDs for source (training) sessions only
            src_domain_ids = torch.tensor(
                [session_to_domain_id[session_name] for session_name in src_session_names],
                dtype=torch.long, device=device
            )
            if rank == 0:
                assert src_domain_ids.min().item() >= 0
                assert src_domain_ids.max().item() < n_domains
            target_domain_ids = F.one_hot(src_domain_ids, num_classes=len(source_days)).float()

            def hinge_loss_logits(logits, target_is_source: bool):
                # logits: [N] (raw logits)
                t = 1.0 if target_is_source else -1.0
                return torch.clamp(1.0 - t * logits, min=0.0)
            def binary_cross_entropy_logits(logits, target_is_source: bool):
                # logits: [N] (raw logits)
                t = 1.0 if target_is_source else 0.0
                return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits) * t, reduction="sum")

            # Domain classifier (train during warmup without GRL)
            if current_lambda_domain > 0 or is_warmup_epoch:
                if truely_mdan:
                    # ============================================================
                    # MDAN (one-vs-on):
                    #   For each head i: positive = domain i (source-i),
                    #                    negative = just target
                    # ============================================================

                    # ---- MDAN: K binary heads; worst-case (max) over heads ----
                    per_head_losses = []
                    dom_losses = []
   

                    # for logging (per-sample normalized)
                    src_loss_sum = torch.tensor(0.0, device=device)
                    src_count    = torch.tensor(0.0, device=device)
                    tgt_loss_sum = torch.tensor(0.0, device=device)
                    tgt_count    = torch.tensor(0.0, device=device)

                    for disc_idx, disc in enumerate(domain_disc):

                        ls_min = ls_max = float('nan')
                        lt_min = lt_max = float('nan')

                        src_mask = (src_domain_ids == disc_idx)
                        m = int(src_mask.sum().item())
                        # ----- source-i term (label=1) -----
                        if src_mask.any():
                            # mask valid T of each sample
                            if args.get('mean_pool_for_discriminator') is False:
                                Bs, Ts, D = src_feat_grl.shape
                                t = torch.arange(Ts, device=device)[None, :].expand(Bs, Ts)
                                adjustedLens_mask = (t < adjustedLens_src[:, None])  # [B, T]
                                sel_s = src_mask[:, None] & adjustedLens_mask
                            else:
                                sel_s = src_mask

                            # mask source i features
                            logits_s = disc(src_feat_grl[sel_s]).view(-1)

                            # src_loss_i_sum = F.binary_cross_entropy_with_logits(
                            #     logits_s, torch.ones_like(logits_s), reduction="sum"
                            # )

                            # hinge instead of binary cross entropy
                            ls_min, ls_max = logits_s.min().item(), logits_s.max().item()
                            if args.get('binary_cross_entropy_loss', False):
                                src_loss_i_ = binary_cross_entropy_logits(logits_s, target_is_source=True)
                            else:
                                src_loss_i_ =  hinge_loss_logits(logits_s, target_is_source=True)
                            if args.get('non_blank_weighting', False):
                                # find the frames that are not "blank" token
                                non_blank_weight_s = src_non_blank_w[sel_s].view(-1).detach()
                                non_blank_weight_s = torch.clamp(non_blank_weight_s, min=0.05, max=1.0)

                                # multiply the loss by the non-blank weight
                                src_loss_i_sum = (non_blank_weight_s * src_loss_i_).sum() 
                                src_weight_sum = non_blank_weight_s.sum() + 1e-8

                                # per head mean loss
                                src_loss_i = src_loss_i_sum / src_weight_sum

                                # logging
                                src_loss_sum += src_loss_i_sum
                                src_count += src_weight_sum
                                
                            else:
                                src_loss_i_sum = src_loss_i_.sum()
                                src_weight_sum = logits_s.numel() 

                                # per-head mean loss used for MDAN max
                                src_loss_i = src_loss_i_sum / src_weight_sum


                                src_loss_sum += src_loss_i_sum
                                src_count    += src_weight_sum
                        else:
                            src_loss_i = torch.tensor(0.0, device=device)                      

                        # ----- target term (label=0) -----  
                        if use_target_loss and (tgt_feat_grl is not None) and m > 0:
                            # m is the number of source samples for this head
                            # we want to sample mm target samples to balance the counts
                            # otherwise, there will always be more target negative samples than source-i positive samples

                            if args.get('mean_pool_for_discriminator') is False:
                                # caveat: this balances # samples, when masked_mean_pool is false, feature is # sample x T, and T varies by sample
                                # Not guaranteed equal total frame counts
                                Bt, Tt, D = tgt_feat_grl.shape
                                mm = min(m, Bt)
                                tgt_mask = torch.randperm(Bt, device=device)[:mm]
                                tgt_feat_grl_sel = tgt_feat_grl[tgt_mask]
                                lens_sel = adjustedLens_tgt[tgt_mask]

                                # mask valid T of each sample
                                tt = torch.arange(Tt, device=device)[None, :].expand(mm, Tt)
                                sel_t = (tt < lens_sel[:, None])

                                # mask same number of samples as source-i
                                logits_t = disc(tgt_feat_grl_sel[sel_t]).view(-1)

                            else:
                                Bt, D = tgt_feat_grl.shape
                                mm = min(m, Bt)
                                tgt_mask = torch.randperm(Bt, device=device)[:mm]
                                tgt_feat_grl_sel = tgt_feat_grl[tgt_mask]
                                logits_t = disc(tgt_feat_grl_sel).view(-1)  
                                      
                            # tgt_loss_i_sum = F.binary_cross_entropy_with_logits(
                            #     logits_t, torch.zeros_like(logits_t), reduction="sum"
                            # )

                            
                            lt_min, lt_max = logits_t.min().item(), logits_t.max().item()

                            if args.get('binary_cross_entropy_loss', False):
                                tgt_loss_i_ = binary_cross_entropy_logits(logits_t, target_is_source=False)
                            else:
                                # hinge instead of binary cross entropy
                                tgt_loss_i_ =  hinge_loss_logits(logits_t, target_is_source=False)
                            
                            if args.get('non_blank_weighting', False):
                                non_blank_weight_t = tgt_non_blank_w[tgt_mask][sel_t].view(-1).detach()
                                non_blank_weight_t = torch.clamp(non_blank_weight_t, min=0.05, max=1.0)

                                # multiply the loss by the non-blank weight
                                tgt_loss_i_sum = (non_blank_weight_t * tgt_loss_i_).sum() 
                                tgt_weight_sum = non_blank_weight_t.sum() + 1e-8

                                # per head mean loss
                                tgt_loss_i = tgt_loss_i_sum / tgt_weight_sum

                                # logging
                                tgt_loss_sum += tgt_loss_i_sum
                                tgt_count += tgt_weight_sum
                            else:
                                tgt_loss_i_sum = tgt_loss_i_.sum()
                                tgt_weight_sum = logits_t.numel()

                                # per head mean loss
                                tgt_loss_i = tgt_loss_i_sum / tgt_weight_sum

                                # logging
                                tgt_loss_sum += tgt_loss_i_sum
                                tgt_count    += tgt_weight_sum

                                # entropy maximization (binary case)
                                # p has shape [B]; compute Bernoulli entropy per-sample
                                # p = torch.sigmoid(logits_t)
                                # entropy = -(
                                #     p * torch.log(p + 1e-8)
                                #     + (1.0 - p) * torch.log(1.0 - p + 1e-8)
                                # )
                                # tgt_loss_i_sum = -entropy.sum()
                        else:
                            tgt_loss_i = torch.tensor(0.0, device=device)
                        if not math.isnan(ls_min):
                            train_src_logits_mins.append(ls_min)
                            train_src_logits_maxs.append(ls_max)
                        if not math.isnan(lt_min):
                            train_tgt_logits_mins.append(lt_min)
                            train_tgt_logits_maxs.append(lt_max)

                        # ----- MDAN objective: worst-case head -----
                        dom_loss_i = 0.5 * (src_loss_i + lambda_tgt *tgt_loss_i)
                        dom_losses.append(dom_loss_i)
                    
                    if len(dom_losses) > 0:
                        # mean over all heads
                        if args.get('dom_loss_type') == 'mean':
                            losses = torch.stack(dom_losses)  
                            dom_loss = losses.mean()
                        elif args.get('dom_loss_type') == 'sum':
                            losses = torch.stack(dom_losses)  
                            dom_loss = losses.mean() * losses.numel()
                        elif args.get('dom_loss_type') == 'max':
                            # when is max better than mean?
                            # only when we want robust worst-case adaptation
                            # however, I observed target_mean_max_logit spikes -> this means one head is very confident

                            # max over all heads
                            dom_loss = torch.stack(dom_losses).max()
                        elif args.get('dom_loss_type') == 'softmax':
                            # softmax with temperature
                            # this will make heads with higher loss, to have higher weight, thus contribute more to dom_loss
                            # thus, heads with lower accuracy will be penalized more with GRL
                            
                            temperature = args.get('temperature', 1.0)
                            losses = torch.stack(dom_losses)              # [K]
                            weights = torch.softmax(losses / temperature, dim=0)
                            if args.get('softmax_w_detach', False) == True:
                                weights = weights.detach()
                            
                            dom_loss = torch.sum(weights * losses)
                        elif args.get('dom_loss_type') == 'softmax_asymmetric':
                            # softmax with temperature
                            # this will make heads with lower loss, to have higher weight, thus contribute more to dom_loss
                            # thus, heads with higher accuracy will be penalized more with GRL
                            # because I observed 5/8 heads stay at 80%, might because GRL didn’t push them hard enough.
                            temperature = args.get('temperature', 1.0) * 5
                            losses = torch.stack(dom_losses)              # [K]
                            weights = torch.softmax(-losses / temperature, dim=0)
                            if args.get('softmax_w_detach', False) == True:
                                weights = weights.detach()
                            
                            dom_loss = torch.sum(weights * losses)
                        elif args.get('dom_loss_type') == 'min':
                            losses = torch.stack(dom_losses)  
                            dom_loss = losses.min()
                        #elif args.get('dom_loss_type') == 'mmd': #TODO

                        else:
                            losses = torch.stack(dom_losses)  
                            dom_loss = losses.mean()
                            raise ValueError(f"Invalid dom_loss_type: {args.get('dom_loss_type')}")

                    else:
                        dom_loss = torch.tensor(0.0, device=device)

                    # Apply domain entropy weight in MDAN path (maximize entropy of target logits; encoder-only gradient)
                    domain_entropy_weight = float(args.get("domain_entropy_weight", 0.0))
                    if domain_entropy_weight > 0 and use_target_loss and (tgt_feat_grl is not None) and len(dom_losses) > 0:
                        req_flags = [p.requires_grad for p in domain_disc.parameters()]
                        for p in domain_disc.parameters():
                            p.requires_grad_(False)
                        ent_terms = []
                        if args.get('mean_pool_for_discriminator') is True:
                            tgt_flat = tgt_feat_grl  # [Bt, D]
                        else:
                            Bt, Tt, _ = tgt_feat_grl.shape
                            tt = torch.arange(Tt, device=device)[None, :].expand(Bt, Tt)
                            adjustedLens_mask_tgt_mdan = (tt < adjustedLens_tgt[:, None])
                            tgt_flat = tgt_feat_grl[adjustedLens_mask_tgt_mdan]
                        for disc in domain_disc:
                            logits_t = disc(tgt_flat).view(-1)
                            p = torch.sigmoid(logits_t)
                            ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
                            ent_terms.append(ent)
                        entropy_loss_mdan = torch.stack(ent_terms).mean()
                        dom_loss = dom_loss - domain_entropy_weight * entropy_loss_mdan
                        for p, req in zip(domain_disc.parameters(), req_flags):
                            p.requires_grad_(req)

                    mdan_loss = task_loss + current_lambda_domain * dom_loss
                    if is_warmup_epoch:
                        # lambda domain is 0 during warmup
                        mdan_loss = task_loss + dom_loss

                    # per-sample normalized losses for logging
                    src_dom_loss = src_loss_sum / (src_count + 1e-8)
                    tgt_dom_loss = tgt_loss_sum / (tgt_count + 1e-8)
                else:
                    # ============================================================
                    # Multiclass domain classifier:
                    #   classify source i vs all other sources + target (single head, CE loss)
                    # ============================================================
                    src_loss_sum = torch.tensor(0.0, device=device)
                    src_count    = torch.tensor(0.0, device=device)
                    tgt_loss_sum = torch.tensor(0.0, device=device)
                    tgt_count    = torch.tensor(0.0, device=device)

                    target_class = n_domains  # sources: [0..n_domains-1], target: n_domains

                    adjustedLens_mask_src = None
                    adjustedLens_mask_tgt = None

                    if args.get('mean_pool_for_discriminator') is True:
                        # ---- source ----
                        src_logits_dom = domain_disc(src_feat_grl)  # [Bs, C]
                        src_labels = src_domain_ids
                        w_src = None
                        if args.get('non_blank_weighting', False):
                            w_src = masked_mean_1d(src_non_blank_w, adjustedLens_src).detach()
                        src_loss_vec = F.cross_entropy(src_logits_dom, src_labels, reduction="none")
                        if args.get('non_blank_weighting', False):
                            src_loss_sum = (w_src * src_loss_vec).sum()
                            src_count = w_src.sum() + 1e-8
                        else:
                            src_loss_sum = src_loss_vec.sum()
                            src_count = src_loss_vec.numel()
                        src_dom_loss = src_loss_sum / (src_count + 1e-8)

                        # ---- target ----
                        if use_target_loss and (tgt_feat_grl is not None):
                            tgt_logits_dom = domain_disc(tgt_feat_grl)  # [Bt, C]
                            tgt_labels = torch.full(
                                (tgt_logits_dom.shape[0],),
                                target_class,
                                device=device,
                                dtype=torch.long,
                            )
                            w_tgt = None
                            if args.get('non_blank_weighting', False):
                                w_tgt = masked_mean_1d(tgt_non_blank_w, adjustedLens_tgt).detach()
                            tgt_loss_vec = F.cross_entropy(tgt_logits_dom, tgt_labels, reduction="none")
                            if args.get('non_blank_weighting', False):
                                tgt_loss_sum = (w_tgt * tgt_loss_vec).sum()
                                tgt_count = w_tgt.sum() + 1e-8
                            else:
                                tgt_loss_sum = tgt_loss_vec.sum()
                                tgt_count = tgt_loss_vec.numel()
                            tgt_dom_loss = tgt_loss_sum / (tgt_count + 1e-8)
                        else:
                            tgt_dom_loss = torch.tensor(0.0, device=device)
                    else:
                        # ---- source (per-frame) ----
                        Bs, Ts, _ = src_feat_grl.shape
                        t = torch.arange(Ts, device=device)[None, :].expand(Bs, Ts)
                        adjustedLens_mask_src = (t < adjustedLens_src[:, None])  # [B, T]
                        src_logits_dom = domain_disc(src_feat_grl[adjustedLens_mask_src])  # [N, C]
                        src_labels = src_domain_ids[:, None].expand(Bs, Ts)[adjustedLens_mask_src]
                        w_src = None
                        if args.get('non_blank_weighting', False):
                            w_src = src_non_blank_w[adjustedLens_mask_src].view(-1).detach()
                            w_src = torch.clamp(w_src, min=0.05, max=1.0)
                        src_loss_vec = F.cross_entropy(src_logits_dom, src_labels, reduction="none")
                        if args.get('non_blank_weighting', False):
                            src_loss_sum = (w_src * src_loss_vec).sum()
                            src_count = w_src.sum() + 1e-8
                        else:
                            src_loss_sum = src_loss_vec.sum()
                            src_count = src_loss_vec.numel()
                        src_dom_loss = src_loss_sum / (src_count + 1e-8)

                        # ---- target (per-frame) ----
                        if use_target_loss and (tgt_feat_grl is not None):
                            Bt, Tt, _ = tgt_feat_grl.shape
                            tt = torch.arange(Tt, device=device)[None, :].expand(Bt, Tt)
                            adjustedLens_mask_tgt = (tt < adjustedLens_tgt[:, None])  # [Bt, Tt]
                            tgt_logits_dom = domain_disc(tgt_feat_grl[adjustedLens_mask_tgt])  # [N, C]
                            tgt_labels = torch.full(
                                (tgt_logits_dom.shape[0],),
                                target_class,
                                device=device,
                                dtype=torch.long,
                            )
                            w_tgt = None
                            if args.get('non_blank_weighting', False):
                                w_tgt = tgt_non_blank_w[adjustedLens_mask_tgt].view(-1).detach()
                                w_tgt = torch.clamp(w_tgt, min=0.05, max=1.0)
                            tgt_loss_vec = F.cross_entropy(tgt_logits_dom, tgt_labels, reduction="none")
                            if args.get('non_blank_weighting', False):
                                tgt_loss_sum = (w_tgt * tgt_loss_vec).sum()
                                tgt_count = w_tgt.sum() + 1e-8
                            else:
                                tgt_loss_sum = tgt_loss_vec.sum()
                                tgt_count = tgt_loss_vec.numel()
                            tgt_dom_loss = tgt_loss_sum / (tgt_count + 1e-8)
                        else:
                            tgt_dom_loss = torch.tensor(0.0, device=device)

                    if use_target_loss and (tgt_feat_grl is not None):
                        dom_loss = 0.5 * (src_dom_loss + tgt_dom_loss)
                    else:
                        dom_loss = src_dom_loss

                    domain_entropy_weight = float(args.get("domain_entropy_weight", 0.0))
                    if domain_entropy_weight > 0:
                        def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
                            p = torch.softmax(logits, dim=1)
                            return -(p * torch.log(p + 1e-8)).sum(dim=1)  # [N]

                        # Freeze discriminator params so entropy term doesn't update disc
                        req_flags = [p.requires_grad for p in domain_disc.parameters()]
                        for p, req in zip(domain_disc.parameters(), req_flags):
                            if req:
                                p.requires_grad_(False)

                        ent_terms = []
                        if args.get('mean_pool_for_discriminator') is True:
                            src_logits_for_ent = domain_disc(src_feat_grl)
                            ent_src = entropy_from_logits(src_logits_for_ent)
                            if args.get('non_blank_weighting', False):
                                w_src_ent = masked_mean_1d(src_non_blank_w, adjustedLens_src).detach()
                                ent_src = (w_src_ent * ent_src).sum() / (w_src_ent.sum() + 1e-8)
                            else:
                                ent_src = ent_src.mean()
                            ent_terms.append(ent_src)

                            if use_target_loss and (tgt_feat_grl is not None):
                                tgt_logits_for_ent = domain_disc(tgt_feat_grl)
                                ent_tgt = entropy_from_logits(tgt_logits_for_ent)
                                if args.get('non_blank_weighting', False):
                                    w_tgt_ent = masked_mean_1d(tgt_non_blank_w, adjustedLens_tgt).detach()
                                    ent_tgt = (w_tgt_ent * ent_tgt).sum() / (w_tgt_ent.sum() + 1e-8)
                                else:
                                    ent_tgt = ent_tgt.mean()
                                ent_terms.append(ent_tgt)
                        else:
                            src_logits_for_ent = domain_disc(src_feat_grl[adjustedLens_mask_src])
                            ent_src = entropy_from_logits(src_logits_for_ent)
                            if args.get('non_blank_weighting', False):
                                w_src_ent = src_non_blank_w[adjustedLens_mask_src].view(-1).detach()
                                w_src_ent = torch.clamp(w_src_ent, min=0.05, max=1.0)
                                ent_src = (w_src_ent * ent_src).sum() / (w_src_ent.sum() + 1e-8)
                            else:
                                ent_src = ent_src.mean()
                            ent_terms.append(ent_src)

                            if use_target_loss and (tgt_feat_grl is not None):
                                tgt_logits_for_ent = domain_disc(tgt_feat_grl[adjustedLens_mask_tgt])
                                ent_tgt = entropy_from_logits(tgt_logits_for_ent)
                                if args.get('non_blank_weighting', False):
                                    w_tgt_ent = tgt_non_blank_w[adjustedLens_mask_tgt].view(-1).detach()
                                    w_tgt_ent = torch.clamp(w_tgt_ent, min=0.05, max=1.0)
                                    ent_tgt = (w_tgt_ent * ent_tgt).sum() / (w_tgt_ent.sum() + 1e-8)
                                else:
                                    ent_tgt = ent_tgt.mean()
                                ent_terms.append(ent_tgt)

                        entropy_loss = -torch.stack(ent_terms).mean() if len(ent_terms) > 0 else torch.tensor(0.0, device=device)

                        # Restore discriminator requires_grad flags
                        for p, req in zip(domain_disc.parameters(), req_flags):
                            p.requires_grad_(req)

                        dom_loss = dom_loss - domain_entropy_weight * entropy_loss # negative, because we want to maximize entropy (logits are computed from GRL features)

                    mdan_loss = task_loss + current_lambda_domain * dom_loss
                    if is_warmup_epoch:
                        # lambda domain is 0 during warmup
                        mdan_loss = task_loss + dom_loss

            else:
                print("Warning: discriminator is not trained")
                mdan_loss = task_loss
                dom_loss = torch.tensor(0.0, device=device, requires_grad=False)
                src_dom_loss = torch.tensor(0.0, device=device, requires_grad=False)
                tgt_dom_loss = torch.tensor(0.0, device=device, requires_grad=False)




            loss = mdan_loss

            # Gradient diagnostics for encoder: task vs domain (after GRL + lambda scaling)
            g_task = float('nan')
            g_dom = float('nan')
            dom_task_grad_ratio = float('nan')
            dom_task_grad_cos = float('nan')
            task_grad_vec = None
            dom_grad_vec = None
            if len(encoder_params) > 0:
                task_grads = torch.autograd.grad(
                    task_loss, encoder_params, retain_graph=True, allow_unused=True
                )
                task_grad_vec = _flatten_grads(encoder_params, task_grads)
                if task_grad_vec.numel() > 0:
                    g_task = task_grad_vec.norm(p=2).item()

                if current_lambda_domain > 0 and dom_loss is not None:
                    dom_term = current_lambda_domain * dom_loss
                    dom_grads = torch.autograd.grad(
                        dom_term, encoder_params, retain_graph=True, allow_unused=True
                    )
                    dom_grad_vec = _flatten_grads(encoder_params, dom_grads)
                    if dom_grad_vec.numel() > 0:
                        g_dom = dom_grad_vec.norm(p=2).item()

                if (
                    task_grad_vec is not None
                    and dom_grad_vec is not None
                    and task_grad_vec.numel() > 0
                    and dom_grad_vec.numel() > 0
                ):
                    dom_task_grad_ratio = g_dom / (g_task + grad_metrics_eps)
                    dot = torch.dot(task_grad_vec, dom_grad_vec)
                    dom_task_grad_cos = (dot / (task_grad_vec.norm() * dom_grad_vec.norm() + grad_metrics_eps)).item()

            if current_lambda_domain <= 0:
                g_dom = 0.0
                dom_task_grad_ratio = 0.0

            train_task_grad_norms.append(g_task)
            train_dom_grad_norms.append(g_dom)
            train_dom_task_grad_ratios.append(dom_task_grad_ratio)
            train_dom_task_grad_coss.append(dom_task_grad_cos)

            use_afn = bool(args.get("use_afn", True))
            afn_weight = float(args.get("afn_weight", 0.01))
            afn_mode = args.get("afn_mode", "safn")  # "hafn" or "safn"
            afn_R = float(args.get("afn_R", 25.0))   # for HAFN
            afn_delta_r = float(args.get("afn_delta_r", 1.0))  # for SAFN

            afn_term = torch.tensor(0.0, device=device)

            if use_afn and afn_weight > 0:
                # IMPORTANT: use unnormalized features for AFN
                # afn_term = afn_term + afn_loss(src_feat, mode=afn_mode, R=afn_R, delta_r=afn_delta_r)
                
                # current sota tgt norm is bigger than src norm, so we apply hafn to src only to expand norm of src
                if use_target_loss and (tgt_feat is not None):
                    afn_term = afn_term + afn_loss(tgt_feat, mode=afn_mode, R=afn_R, delta_r=afn_delta_r)

                # Final loss (you already have task_loss + lambda_domain*dom_loss in mdan_loss)
                loss = mdan_loss + afn_weight * afn_term

            # Backward
            loss.backward()

            # Apply warmup: lambda_domain = 0 during warmup epochs
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), grad_clip)

            # Step optimizers separately
            model_optimizer.step()

            #===============
            # Metrics
            #===============

            # Accuracy proxy: pick the head with max logit on source examples
            with torch.no_grad():
                src_feat_grl = src_feat_grl.detach()
                tgt_feat_grl = tgt_feat_grl.detach()


                if truely_mdan:
                    # source accuracy: check each sample on its *own* head
                    if args.get('mean_pool_for_discriminator') is True:
                        src_logits_own = torch.empty_like(src_domain_ids, dtype=torch.float, device=device)
                        for i, disc in enumerate(domain_disc):
                            m = (src_domain_ids == i)  # [B]
                            if m.any():
                                src_logits_own[m] = disc(src_feat_grl[m]).view(-1)  # [m]
                        train_dom_correct_mdan += (src_logits_own > 0).sum().item()
                        train_dom_total_mdan   += src_logits_own.numel()
                    else:
                        B, T, _ = src_feat_grl.shape
                        t = torch.arange(T, device=device)[None, :].expand(B, T)
                        adjustedLens_mask = (t < adjustedLens_src[:, None])   # [B, T]
                        src_logits_own_bt = torch.zeros(B, T, dtype=torch.float, device=device)
                        for disc_idx, disc in enumerate(domain_disc):
                            m = (src_domain_ids == disc_idx)
                            if m.any():
                                sel_s = m[:, None] & adjustedLens_mask
                                src_logits_own_bt[sel_s] = disc(src_feat_for_discriminator[sel_s]).view(-1)
                        src_logits_valid = src_logits_own_bt[adjustedLens_mask]
                        train_dom_correct_mdan += (src_logits_valid > 0).sum().item()
                        train_dom_total_mdan   += src_logits_valid.numel()


                # per head domain accuracy
                has_tgt = (use_target_loss and (tgt_feat_disc is not None))
                if args.get('mean_pool_for_discriminator') is True:
                    adjustedLens_mask_src = None
                else:
                    B, T, _ = src_feat_grl.shape
                    t = torch.arange(T, device=device)[None, :].expand(B, T)
                    adjustedLens_mask_src = (t < adjustedLens_src[:, None])   # [B, T]

                for i, disc in enumerate(domain_disc):
                    # --- source-i ---
                    m = (src_domain_ids == i)
                    if m.any():
                        if args.get('mean_pool_for_discriminator') is True:
                            logits_s = disc(src_feat_grl[m]).view(-1)   # [m]
                        else:
                            sel_s = m[:, None] & adjustedLens_mask_src
                            logits_s = disc(src_feat_grl[sel_s]).view(-1) # TODO: check if this is correct
                        head_src_correct[i] += (logits_s > 0).sum()
                        head_src_total[i]   += logits_s.numel()

                    # --- target (all) ---
                    if has_tgt:
                        if args.get('mean_pool_for_discriminator') is True:
                            Bt = tgt_final.shape[0]  # tgt_final: [Bt, D]
                            mm = min(int((src_domain_ids == i).sum().item()), Bt)   # match m for head i (same as training)
                            if mm > 0:
                                tgt_mask = torch.randperm(Bt, device=device)[:mm]
                                tgt_feat_sel = tgt_feat_grl.detach()[tgt_mask]       # [mm, D]
                                logits_t = disc(tgt_feat_sel).view(-1)                # [mm]
                                head_tgt_correct[i] += (logits_t < 0).sum()
                                head_tgt_total[i]   += logits_t.numel()
                        else:
                            Bt, Tt, D = tgt_feat_grl.shape
                            mm = min(int((src_domain_ids == i).sum().item()), Bt)
                            if mm > 0:
                                tgt_mask = torch.randperm(Bt, device=device)[:mm]
                                tgt_feat_sel = tgt_feat_grl.detach()[tgt_mask]       # [mm, Tt, D]
                                lens_sel = adjustedLens_tgt[tgt_mask]                 # [mm]

                                tt = torch.arange(Tt, device=device)[None, :].expand(mm, Tt)
                                sel_t = (tt < lens_sel[:, None])                      # [mm, Tt]

                                logits_t = disc(tgt_feat_sel[sel_t]).view(-1)         # [N_valid_frames]
                                head_tgt_correct[i] += (logits_t < 0).sum()
                                head_tgt_total[i]   += logits_t.numel()

                if use_target_loss and (tgt_feat_grl is not None):
                    if args.get('mean_pool_for_discriminator') is True:
                        # [B, K] directly
                        all_tgt_logits = torch.cat(
                            [disc(tgt_feat_grl.detach()).view(-1, 1) for disc in domain_disc],
                            dim=1
                        )
                    else:
                        Bt, Tt, D = tgt_feat_grl.shape
                        tt = torch.arange(Tt, device=device)[None, :].expand(Bt, Tt)
                        tgt_valid = (tt < adjustedLens_tgt[:, None])                  # [Bt, Tt]
                        denom = tgt_valid.sum(dim=1).clamp_min(1).float()             # [Bt]

                        per_head_logits = []
                        for disc in domain_disc:
                            # fill per-frame logits only on valid frames
                            logits_bt = torch.zeros(Bt, Tt, dtype=torch.float, device=device)
                            logits_valid = disc(tgt_feat_grl.detach()[tgt_valid]).view(-1)
                            logits_bt[tgt_valid] = logits_valid
                            logits_mean = logits_bt.sum(dim=1) / denom                # [Bt]
                            per_head_logits.append(logits_mean.view(-1, 1))

                        all_tgt_logits = torch.cat(per_head_logits, dim=1)            # [Bt, K]

                    # target metrics (1): all head predict as not source-i (per SAMPLE)
                    tgt_none_rate = (all_tgt_logits < 0).all(dim=1).float().mean().item()
                    train_tgt_none_rates.append(tgt_none_rate)

                    # target metrics (2): mean(max target logits across all i heads)
                    tgt_max_logit = all_tgt_logits.max(dim=1).values.mean().item()
                    train_tgt_max_logits.append(tgt_max_logit)

                    # target metrics (3): entropy of target logits
                    tgt_probs = torch.sigmoid(all_tgt_logits)
                    tgt_entropy = (
                        -(
                            tgt_probs * torch.log(tgt_probs + 1e-8)
                            + (1.0 - tgt_probs) * torch.log(1.0 - tgt_probs + 1e-8)
                        ).mean()
                    ).item()
                    train_tgt_entropies.append(tgt_entropy)

                    # target metrics (4): fraction of target logits near boundary
                    tgt_frac_boundary = (all_tgt_logits.abs() < 1.0).float().mean().item()
                    train_tgt_frac_boundaries.append(tgt_frac_boundary)
            if current_lambda_domain > 0 or is_warmup_epoch or detach_discriminator_from_encoder:
                disc_optimizer.step()
    
            # Track losses and accuracy
            train_task_losses.append(float(task_loss.item()))
            train_mdan_losses.append(float(mdan_loss.item()))
            train_src_dom_losses.append(float(src_dom_loss.item()))
            train_tgt_dom_losses.append(float(tgt_dom_loss.item()))
            train_total_losses.append(float(loss.item()))
            global_step += 1

        with torch.no_grad():
            # Compute average training losses
            avg_train_task_loss = np.mean(train_task_losses) if len(train_task_losses) > 0 else 0.0
            avg_train_mdan_loss = np.mean(train_mdan_losses) if len(train_mdan_losses) > 0 else 0.0
            avg_train_src_dom_loss = np.mean(train_src_dom_losses) if len(train_src_dom_losses) > 0 else 0.0
            avg_train_tgt_dom_loss = np.mean(train_tgt_dom_losses) if len(train_tgt_dom_losses) > 0 else 0.0
            if truely_mdan:
                avg_train_dom_accuracy = train_dom_correct_mdan / train_dom_total_mdan if train_dom_total_mdan > 0 else 0.0
            else:
                avg_train_dom_accuracy = train_dom_correct / train_dom_total if train_dom_total > 0 else 0.0
            avg_train_total_loss = np.mean(train_total_losses) if len(train_total_losses) > 0 else 0.0
            avg_train_encoder_grad_norm = np.mean(train_encoder_grad_norms) if len(train_encoder_grad_norms) > 0 else 0.0
            avg_train_disc_grad_norm = np.mean(train_disc_grad_norms) if len(train_disc_grad_norms) > 0 else 0.0
            avg_task_grad_norm = float(np.nanmean(train_task_grad_norms)) if len(train_task_grad_norms) > 0 else float('nan')
            avg_dom_grad_norm = float(np.nanmean(train_dom_grad_norms)) if len(train_dom_grad_norms) > 0 else float('nan')
            avg_dom_task_grad_ratio = float(np.nanmean(train_dom_task_grad_ratios)) if len(train_dom_task_grad_ratios) > 0 else float('nan')
            avg_dom_task_grad_cos = float(np.nanmean(train_dom_task_grad_coss)) if len(train_dom_task_grad_coss) > 0 else float('nan')
            epoch_alpha_values = train_alpha_values[-1] if len(train_alpha_values) > 0 else 0.0
            avg_train_tgt_none_rate = float(np.mean(train_tgt_none_rates)) if len(train_tgt_none_rates) > 0 else 0.0
            avg_train_tgt_max_logit = float(np.mean(train_tgt_max_logits)) if len(train_tgt_max_logits) > 0 else float('nan')
            avg_train_tgt_entropy = float(np.mean(train_tgt_entropies)) if len(train_tgt_entropies) > 0 else float('nan')
            avg_train_tgt_frac_boundary = float(np.mean(train_tgt_frac_boundaries)) if len(train_tgt_frac_boundaries) > 0 else float('nan')
            avg_train_src_logit_min = float(np.mean(train_src_logits_mins)) if len(train_src_logits_mins) > 0 else float('nan')
            avg_train_src_logit_max = float(np.mean(train_src_logits_maxs)) if len(train_src_logits_maxs) > 0 else float('nan')
            avg_train_tgt_logit_min = float(np.mean(train_tgt_logits_mins)) if len(train_tgt_logits_mins) > 0 else float('nan')
            avg_train_tgt_logit_max = float(np.mean(train_tgt_logits_maxs)) if len(train_tgt_logits_maxs) > 0 else float('nan')
            
            model.eval()
            probe_mean_dist = float('nan')
            probe_cov_dist = float('nan')
            if (not is_distributed) or (rank == 0):
                probe_src_mean, probe_src_cov = _compute_probe_mean_and_cov(probe_src_loader)
                probe_eval_mean, probe_eval_cov = _compute_probe_mean_and_cov(probe_eval_loader)
                if (probe_src_mean is not None) and (probe_eval_mean is not None):
                    probe_mean_dist = torch.norm(probe_src_mean - probe_eval_mean, p=2).item()
                if (probe_src_cov is not None) and (probe_eval_cov is not None):
                    probe_cov_dist = torch.norm(probe_src_cov - probe_eval_cov, p='fro').item()
            

            # Synchronize across processes if distributed
            if is_distributed:
                avg_train_task_loss_tensor = torch.tensor(avg_train_task_loss, device=device)
                avg_train_mdan_loss_tensor = torch.tensor(avg_train_mdan_loss, device=device)
                avg_train_src_dom_loss_tensor = torch.tensor(avg_train_src_dom_loss, device=device)
                avg_train_tgt_dom_loss_tensor = torch.tensor(avg_train_tgt_dom_loss, device=device)
                avg_train_encoder_grad_norm_tensor = torch.tensor(avg_train_encoder_grad_norm, device=device)
                avg_train_disc_grad_norm_tensor = torch.tensor(avg_train_disc_grad_norm, device=device)
                correct_t = torch.tensor(train_dom_correct if not truely_mdan else train_dom_correct_mdan, device=device, dtype=torch.long)
                total_t   = torch.tensor(train_dom_total if not truely_mdan else train_dom_total_mdan, device=device, dtype=torch.long)
                t = torch.tensor([avg_train_tgt_none_rate, avg_train_tgt_max_logit, avg_train_tgt_entropy, avg_train_tgt_frac_boundary, 1.0], device=device)  # [none_rate, max_logit, entropy, frac_boundary, count]
                logit_stats = torch.tensor([
                    avg_train_src_logit_min, avg_train_src_logit_max,
                    avg_train_tgt_logit_min, avg_train_tgt_logit_max,
                    1.0
                ], device=device)
                
                
               
                dist.all_reduce(avg_train_mdan_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_task_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_src_dom_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_tgt_dom_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_encoder_grad_norm_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_disc_grad_norm_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                dist.all_reduce(logit_stats, op=dist.ReduceOp.SUM)
                dist.all_reduce(head_src_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(head_src_total,   op=dist.ReduceOp.SUM)
                dist.all_reduce(head_tgt_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(head_tgt_total,   op=dist.ReduceOp.SUM)

                avg_train_task_loss = avg_train_task_loss_tensor.item() / world_size
                avg_train_mdan_loss = avg_train_mdan_loss_tensor.item() / world_size
                avg_train_src_dom_loss = avg_train_src_dom_loss_tensor.item() / world_size
                avg_train_tgt_dom_loss = avg_train_tgt_dom_loss_tensor.item() / world_size
                # Fix: Check for division by zero in distributed training
                avg_train_dom_accuracy = (correct_t.float() / total_t.float()).item() if total_t.item() > 0 else 0.0
                avg_train_encoder_grad_norm = avg_train_encoder_grad_norm_tensor.item() / world_size
                avg_train_disc_grad_norm = avg_train_disc_grad_norm_tensor.item() / world_size
                denom = t[4] + 1e-8  # world_size
                avg_train_tgt_none_rate = (t[0] / denom).item()
                avg_train_tgt_max_logit = (t[1] / denom).item()
                avg_train_tgt_entropy = (t[2] / denom).item()
                avg_train_tgt_frac_boundary = (t[3] / denom).item()
                logit_denom = logit_stats[4] + 1e-8
                avg_train_src_logit_min = (logit_stats[0] / logit_denom).item()
                avg_train_src_logit_max = (logit_stats[1] / logit_denom).item()
                avg_train_tgt_logit_min = (logit_stats[2] / logit_denom).item()
                avg_train_tgt_logit_max = (logit_stats[3] / logit_denom).item()
                
            # Compute per-head accuracies (global)

            eps = 1e-8
            head_src_acc = (head_src_correct / (head_src_total + eps)).detach().cpu().numpy()
            head_tgt_acc = (head_tgt_correct / (head_tgt_total + eps)).detach().cpu().numpy()

            # Optional: set NaN where head had no samples
            head_src_total_np = head_src_total.detach().cpu().numpy()
            head_tgt_total_np = head_tgt_total.detach().cpu().numpy()
            head_src_acc = np.where(head_src_total_np > 0, head_src_acc, np.nan)
            head_tgt_acc = np.where(head_tgt_total_np > 0, head_tgt_acc, np.nan)

            head_bal_acc = 0.5 * (head_src_acc + head_tgt_acc)
            head_bal_acc_weighted = (
                (head_src_correct + head_tgt_correct) / (head_src_total + head_tgt_total + eps)
            ).detach().cpu().numpy()

            mean_head_bal_acc = float(np.nanmean(head_bal_acc))
            min_head_bal_acc  = float(np.nanmin(head_bal_acc))
            max_head_bal_acc  = float(np.nanmax(head_bal_acc))
            mean_head_bal_acc_weighted = float(np.nanmean(head_bal_acc_weighted))
            min_head_bal_acc_weighted  = float(np.nanmin(head_bal_acc_weighted))
            max_head_bal_acc_weighted  = float(np.nanmax(head_bal_acc_weighted))           
            
            # Compute train CER - only on rank 0 using full dataset (no sampler) to mimic single GPU
            model.eval()
            train_total_edit_distance = 0
            train_total_seq_length = 0
            train_vowel_err = 0
            train_vowel_count = 0
            train_consonant_err = 0
            train_consonant_count = 0
            train_per_phone_err = np.zeros(len(phoneme_to_id), dtype=np.int64)
            train_per_phone_count = np.zeros(len(phoneme_to_id), dtype=np.int64)
            
            if not is_distributed or rank == 0:
                # Sample a few batches from full training data (no DistributedSampler)
                for train_batch_idx, train_batch in enumerate(trainMetricsLoader):
                    if train_batch_idx >= 10:  # Sample first 10 batches for train CER (faster)
                        break

                    X_train, X_len_train, y_train, y_len_train, dayIdx_train, y2_train, y2_len_train = unpack_batch_5or7(train_batch)
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    X_len_train = X_len_train.to(device)
                    y_len_train = y_len_train.to(device) 
                    dayIdx_train = dayIdx_train.to(device)
                    if y2_train is not None:
                        y2_train = y2_train.to(device)
                    if y2_len_train is not None:
                        y2_len_train = y2_len_train.to(device)
                    
                    adjustedLens_train = base_model.compute_length(X_len_train)
                    pred_train = model(X_train, X_len_train, dayIdx_train, return_rep=False)
                    
                    for iterIdx in range(pred_train.shape[0]):
                        decodedSeq_train = torch.argmax(
                            pred_train[iterIdx, 0:adjustedLens_train[iterIdx], :], dim=-1
                        )
                        decodedSeq_train = torch.unique_consecutive(decodedSeq_train, dim=-1)
                        decodedSeq_train = decodedSeq_train.cpu().detach().numpy()
                        decodedSeq_train = np.array([i for i in decodedSeq_train if i != 0])
                        
                        trueSeq_train = np.array(
                            y_train[iterIdx][0:y_len_train[iterIdx]].cpu().detach()
                        )
                        
                        matcher = SequenceMatcher(
                            a=trueSeq_train.tolist(),
                            b=decodedSeq_train.tolist()
                        )
                        train_total_edit_distance += matcher.distance()
                        train_total_seq_length += len(trueSeq_train)
                        (
                            train_vowel_err,
                            train_vowel_count,
                            train_consonant_err,
                            train_consonant_count,
                        ) = _update_cer_breakdown(
                            trueSeq_train,
                            decodedSeq_train,
                            train_per_phone_err,
                            train_per_phone_count,
                            train_vowel_err,
                            train_vowel_count,
                            train_consonant_err,
                            train_consonant_count,
                        )
                
                train_cer = (
                    train_total_edit_distance / train_total_seq_length
                    if train_total_seq_length > 0 else float('nan')
                )
                train_vowel_cer = (
                    train_vowel_err / train_vowel_count
                    if train_vowel_count > 0 else float('nan')
                )
                train_consonant_cer = (
                    train_consonant_err / train_consonant_count
                    if train_consonant_count > 0 else float('nan')
                )
                train_per_phone_cer = np.full(len(phoneme_to_id), np.nan, dtype=np.float64)
                nonzero_mask = train_per_phone_count > 0
                train_per_phone_cer[nonzero_mask] = (
                    train_per_phone_err[nonzero_mask] / train_per_phone_count[nonzero_mask]
                )
            else:
                train_cer = float('nan')
                train_vowel_cer = float('nan')
                train_consonant_cer = float('nan')
                train_per_phone_cer = np.full(len(phoneme_to_id), np.nan, dtype=np.float64)
            
            # ==============================    
            # Evaluate on test set (only on rank 0 to avoid deadlock)
            # ==============================
            eval_loss = float('nan')
            eval_cer = float('nan')
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            eval_vowel_err = 0
            eval_vowel_count = 0
            eval_consonant_err = 0
            eval_consonant_count = 0
            eval_per_phone_err = np.zeros(len(phoneme_to_id), dtype=np.int64)
            eval_per_phone_count = np.zeros(len(phoneme_to_id), dtype=np.int64)

            model.eval()
            
            if (not is_distributed) or (rank == 0):
                print(f"Evaluating on test set (rank {rank})")  
                for batch in tqdm(eval_testLoader, desc="Evaluating", leave=False):
                    # Unpack batch (handles both 5 and 7 item batches)
                    X, X_len, y, y_len, dayIdx, y2, y2_len = unpack_batch_5or7(batch)
                    
                    if args.get('maxDay') is not None:
                        dayIdx.fill_(args['maxDay'])
                    
                    X = X.to(device)
                    y = y.to(device)
                    X_len = X_len.to(device)
                    y_len = y_len.to(device)
                    dayIdx = dayIdx.to(device)
                    
                    adjustedLens = base_model.compute_length(X_len)
                    
                    pred = model(X, X_len, dayIdx, return_rep=False)
                   
                    loss = forward_ctc(pred, adjustedLens, y, y_len)
                    
                    allLoss.append(loss.item())
                    
                    # CER computation
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            pred[iterIdx, 0:adjustedLens[iterIdx], :], dim=-1
                        )
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])
                        
                        trueSeq = np.array(
                            y[iterIdx][0:y_len[iterIdx]].cpu().detach()
                        )
                        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)
                        (
                            eval_vowel_err,
                            eval_vowel_count,
                            eval_consonant_err,
                            eval_consonant_count,
                        ) = _update_cer_breakdown(
                            trueSeq,
                            decodedSeq,
                            eval_per_phone_err,
                            eval_per_phone_count,
                            eval_vowel_err,
                            eval_vowel_count,
                            eval_consonant_err,
                            eval_consonant_count,
                        )
                
                eval_loss = np.mean(allLoss) if allLoss else 0.0
                eval_cer = (total_edit_distance / total_seq_length
                           if total_seq_length > 0 else float('nan'))
                eval_vowel_cer = (
                    eval_vowel_err / eval_vowel_count
                    if eval_vowel_count > 0 else float('nan')
                )
                eval_consonant_cer = (
                    eval_consonant_err / eval_consonant_count
                    if eval_consonant_count > 0 else float('nan')
                )
                eval_per_phone_cer = np.full(len(phoneme_to_id), np.nan, dtype=np.float64)
                eval_nonzero_mask = eval_per_phone_count > 0
                eval_per_phone_cer[eval_nonzero_mask] = (
                    eval_per_phone_err[eval_nonzero_mask] / eval_per_phone_count[eval_nonzero_mask]
                )
                print(f"Evaluated on test set (rank {rank}) finished")
            else:
                # other ranks don't eval
                eval_loss = float('nan')
                eval_cer = float('nan')
                eval_vowel_cer = float('nan')
                eval_consonant_cer = float('nan')
                eval_per_phone_cer = np.full(len(phoneme_to_id), np.nan, dtype=np.float64)

        current_lr = model_optimizer.param_groups[0]['lr']
        current_disc_lr = disc_optimizer.param_groups[0]['lr']
        
        endTime = time.time()
        elapsed_time = endTime - epoch_start_time
        time_per_batch = elapsed_time / num_batches if num_batches > 0 else 0.0
        
        # Print epoch summary (only on rank 0)
        if not is_distributed or rank == 0:
            warmup_status = f" (WARMUP: lambda=0)" if epoch < dann_warmup_epochs else ""
            msg = (
                f"Epoch {epoch+1}, train task_loss: {avg_train_task_loss:>7f}, "
                f"train mdan_loss: {avg_train_mdan_loss:>7f} "
                f"(src: {avg_train_src_dom_loss:>7f}"
            )
            if use_target_loss:
                msg += f", tgt: {avg_train_tgt_dom_loss:>7f}"
            msg += f"){warmup_status}, train cer: {train_cer:>7f}"
            if eval_testLoader is not None:
                msg += f", eval task_loss: {eval_loss:>7f}, eval cer: {eval_cer:>7f}"
            msg += f", lr: {current_lr:.6f} (disc: {current_disc_lr:.6f}), time/batch: {time_per_batch:>7.3f}"
            print(msg)
            print(f"   Domain accuracy: {avg_train_dom_accuracy:.4f} | "
                  f"Rule of thumb: if domain accuracy stays near chance but task improves → good. "
                  f"If task degrades early → too much domain pressure too soon.")
        
        # Log to wandb (only on rank 0)
        if wandb.run is not None and (not is_distributed or rank == 0):
            log_dict = {
                "train/ctc_loss": avg_train_task_loss,
                "train/mdan_loss (ctc + lambda * (src + tgt))": avg_train_mdan_loss,
                "train/source_domain_loss": avg_train_src_dom_loss,
                "train/target_domain_loss (mean BCE over (head, target-sample) pairs)": avg_train_tgt_dom_loss,
                "train/domain_accuracy (fraction of samples where discriminator says “this is source-i”), ↓ toward ~0.5": avg_train_dom_accuracy,
                "train/total_loss (same as mdan loss)": avg_train_total_loss,
                "train/cer": train_cer,
                "train/vowel_cer": train_vowel_cer,
                "train/consonant_cer": train_consonant_cer,
                "train/learning_rate": current_lr,
                "train/encoder_grad_norm": avg_train_encoder_grad_norm,
                "train/disc_grad_norm": avg_train_disc_grad_norm,
                "train/g_task_norm": avg_task_grad_norm,
                "train/g_dom_norm": avg_dom_grad_norm,
                "train/dom_task_grad_ratio": avg_dom_task_grad_ratio,
                "train/dom_task_grad_cos": avg_dom_task_grad_cos,
                "train/alpha": epoch_alpha_values,
                "train/target_none_rate (P(all heads predict “not source-i”)) ↓ from ~1.0 → moderate plateau (not 0)": avg_train_tgt_none_rate,
                "train/target_mean_max_logit": avg_train_tgt_max_logit,
                "train/target_logit_entropy": avg_train_tgt_entropy,
                "train/target_frac_near_boundary(|logit|<1)": avg_train_tgt_frac_boundary,
                "train/src_logit_min": avg_train_src_logit_min,
                "train/src_logit_max": avg_train_src_logit_max,
                "train/target_logit_min": avg_train_tgt_logit_min,
                "train/target_logit_max": avg_train_tgt_logit_max,
                "train/head_balanced_acc_mean": mean_head_bal_acc,
                "train/head_balanced_acc_min": min_head_bal_acc,
                "train/head_balanced_acc_max": max_head_bal_acc,
                "train/head_balanced_acc_weighted_mean": mean_head_bal_acc_weighted,
                "train/head_balanced_acc_weighted_min": min_head_bal_acc_weighted,
                "train/head_balanced_acc_weighted_max": max_head_bal_acc_weighted,
                "train/probe_mean_dist (on mean pooled feat)": probe_mean_dist,
                "train/probe_cov_dist (on mean pooled feat)": probe_cov_dist,
                "epoch": epoch + 1,
            }
            for i in range(K):
                log_dict[f"train/head_{i}_src_acc_epoch"] = float(head_src_acc[i])
                log_dict[f"train/head_{i}_tgt_acc_epoch"] = float(head_tgt_acc[i])
                log_dict[f"train/head_{i}_bal_acc_epoch"] = float(head_bal_acc[i])
            for phone in PHONE_DEF_SIL:
                pid = phoneme_to_id[phone]
                log_dict[f"train/cer_phoneme/{phone}"] = float(train_per_phone_cer[pid])
                
            if eval_testLoader is not None:
                log_dict["eval/ctc_loss"] = eval_loss
                log_dict["eval/cer"] = eval_cer
                log_dict["eval/vowel_cer"] = eval_vowel_cer
                log_dict["eval/consonant_cer"] = eval_consonant_cer
                for phone in PHONE_DEF_SIL:
                    pid = phoneme_to_id[phone]
                    log_dict[f"eval/cer_phoneme/{phone}"] = float(eval_per_phone_cer[pid])
            wandb.log(log_dict)
        
        # Save checkpoints (only on rank 0)
        if not is_distributed or rank == 0:
            output_dir = args.get('outputDir', './outputs')
            # Save best model based on CER
            if len(testCER) > 0 and eval_cer < np.min(testCER[:-1] if len(testCER) > 1 else [float('inf')]):
                model_state = (
                    model.module.state_dict()
                    if is_distributed and hasattr(model, 'module')
                    else model.state_dict()
                )
                torch.save(model_state, output_dir + "/modelWeights")
                torch.save(model_optimizer.state_dict(), output_dir + "/optimizer")
                torch.save(disc_optimizer.state_dict(), output_dir + "/disc_optimizer")
                torch.save(scheduler.state_dict(), output_dir + "/scheduler")
            
            # Save best model based on loss
            if len(testLoss) > 0 and eval_loss < np.min(testLoss[:-1] if len(testLoss) > 1 else [float('inf')]):
                model_state = (
                    model.module.state_dict()
                    if is_distributed and hasattr(model, 'module')
                    else model.state_dict()
                )
                torch.save(model_state, output_dir + "/modelWeights_ctc")
             
        testLoss.append(eval_loss)
        testCER.append(eval_cer)
        
        # Save training stats
        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)
        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)

        # Step schedulers (both use same scheduler for now, but could be separate)
        scheduler.step()
        # Update discriminator optimizer LR to match model LR * multiplier
        # This ensures the multiplier is maintained even as LR decays
        current_lr_after_step = model_optimizer.param_groups[0]['lr']
        for param_group in disc_optimizer.param_groups:
            param_group['lr'] = current_lr_after_step * dann_lr_multiplier
        
    
    # Only finish wandb on rank 0 (where it was initialized)
    if (not is_distributed or rank == 0) and wandb.run is not None:
        wandb.finish()
    return
