import os
import pickle
import time

from edit_distance import SequenceMatcher
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import getDatasetLoaders, getDatasetLoadersInterleaved
import torch.nn.functional as F
from .loss import forward_ctc, forward_cr_ctc, future_prediction_loss, phone_contrastive_loss, ctc_run_alignment_phone_ids, cross_trial_phone_contrastive_loss, forward_ctc_ntp
from .hi_longformer import compute_hierarchical_ctc_loss
from typing import Tuple, Optional


import wandb

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

def trainModel(args, model):
    
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

    # Check if interleaved dataset should be used
    use_interleaved = args.get('use_interleaved_dataset', False)
    interleave_step = args.get('interleave_step', 10)  # Default: 10% eval (every 10th sample)
    
    if use_interleaved:
        trainLoader, testLoader, loadedData = getDatasetLoadersInterleaved(
            args["datasetPath"],
            args["batchSize"],
            args['restricted_days'], 
            args['ventral_6v_only'],
            interleave_step=interleave_step
        )
    else:
        trainLoader, testLoader, loadedData = getDatasetLoaders(
            args["datasetPath"],
            args["batchSize"],
            args['restricted_days'], 
            args['ventral_6v_only']
        )
    
    # Create a separate loader for train metrics (full dataset, no sampler) to mimic single GPU
    # This ensures train CER is computed on the same data as single GPU mode
    trainMetricsLoader = trainLoader  # Full loader for metrics computation only
    
    # Wrap train loader with DistributedSampler if using distributed training
    train_sampler = None
    test_sampler = None
    if is_distributed:

        batch_size_per_gpu = args["batchSize"] // world_size
        if args["batchSize"] % world_size != 0:
            print(f"⚠️  Warning: batch_size {args['batchSize']} is not divisible by {world_size} GPUs")
            print(f"   Using {batch_size_per_gpu} samples per GPU (total effective batch size: {batch_size_per_gpu * world_size})")
        
        train_sampler = DistributedSampler(
            trainLoader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        trainLoader = torch.utils.data.DataLoader(
            trainLoader.dataset,
            batch_size=batch_size_per_gpu,  # Per GPU batch size
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=trainLoader.collate_fn,
        )
        
        print(f"✅ Using DistributedSampler for training and test data (rank {rank}/{world_size-1})")
        print(f"   Batch size per GPU: {batch_size_per_gpu} (total effective: {batch_size_per_gpu * world_size})")
        print(f"   Train metrics will be computed on full dataset (rank 0 only) to match single GPU behavior")
    
    
    # Watch the model (only if wandb is initialized and on rank 0)
    if wandb.run is not None and (not is_distributed or rank == 0):
        wandb.watch(model.module if is_distributed and hasattr(model, 'module') else model, log="all")  # Logs gradients, parameters, and gradients histograms

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

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

    opt_params = model.parameters()
    
    if args.get('AdamW', False):
        print("USING ADAMW")
        optimizer = torch.optim.AdamW(
            opt_params, 
            lr=float(args['lrStart']), 
            weight_decay=float(args['l2_decay']), 
            betas=(float(args['beta1']), float(args['beta2'])),
        )
    else:
        if args.get('SOAP', False):
            print("USING SOAP")
            from .soap import SOAP
            optimizer = SOAP(
                opt_params,
                lr=float(args['lrStart']),
                betas=(0.95, 0.95),
                weight_decay=float(args['l2_decay']),
                precondition_frequency=int(args.get('precondition_frequency', 10)),
            )
        else:
            print("USING VANILLA ADAM")
            optimizer = torch.optim.Adam(
                opt_params,
                lr=float(args["lrStart"]),
                betas=(0.9, 0.999),
                eps=0.1,
                weight_decay=float(args["l2_decay"]),
            )
    
    if args.get('learning_scheduler', 'None') == 'multistep': 
        print("Multistep scheduler")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=float(args['gamma']),
        )
        
    elif args.get('learning_scheduler', 'None') == 'cosine':
        print("Cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args['n_epochs']),     # Total epochs to decay over
            eta_min=float(args.get('eta_min', 1e-6))    # Final learning rate
        )
            
    elif args.get('learning_scheduler', 'None') == 'warmcosine':
        print("Warm Cosine Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(args.get('T_0', 10)),       # first cosine decay cycle
            T_mult=int(args.get('T_mult', 2)),   # multiplier
            eta_min=float(args.get('eta_min', 1e-6))
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
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=args['device']))
        
        scheduler_path = os.path.join(args['load_pretrained_model'], 'scheduler')
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=args['device']))
        print(f"Loaded optimizer and scheduler state from {args['load_pretrained_model']}")
        
    # --train--
    testLoss = []
    testCER = []
    testCER2 = []
    startTime = time.time()
    train_loss = []
    train_kl_loss = []
    train_deep_loss_list = []
    train_future_pred_losses = []
    train_phone_contrastive_losses = []
    train_ctc_loss = []
    train_ntp_losses = []

    # detect if this is a deep-CTC longformer (works with DDP)
    base_model = model.module if is_distributed and hasattr(model, "module") else model
    is_deep_ctc_model = (
        getattr(base_model, "has_deep_ctc", False) or
        base_model.__class__.__name__ == "DeepCTCLocalGlobalViT_Phoneme"
    )
    is_hierarchical_model = base_model.__class__.__name__ == "HiLocalGlobalViT_Phoneme"
    deep_ctc_weight = args.get("deep_ctc_weight", 0.3)
    
    for epoch in range(args["start_epoch"], args['n_epochs']):
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()  # Track time for this epoch
        train_ctc_loss = []
        train_loss = []
        train_kl_loss = []  # Initialize train_kl_loss for each epoch
        train_deep_loss_list = []
        train_hier_phone_losses = []
        train_hier_broad_losses = []
        train_future_pred_losses = []
        train_phone_contrastive_losses = []
        train_ntp_losses = []
        model.train()

        # Track number of batches for time calculation
        num_batches = 0
        
        # Only show progress bar on rank 0 to avoid interference in distributed training
        progress_bar = tqdm(trainLoader, desc="Training", disable=(is_distributed and rank != 0))
        for batch_idx, batch in enumerate(progress_bar):
            num_batches += 1
           
            # Initialize variables for phone contrastive loss
            rep_for_contrast = None
            logits_for_contrast = None
           
            # Base case: always unpack the first 5
            X, X_len, y, y_len, dayIdx, y2, y2_len = unpack_batch_5or7(batch)

            # Send to device
            X      = X.to(args["device"])
            y      = y.to(args["device"])
            X_len  = X_len.to(args["device"])
            y_len  = y_len.to(args["device"])
            dayIdx = dayIdx.to(args["device"])

            have_second = False
            # if have_second:
            #     y2, y2_len = batch[5], batch[6]
            #     y2     = y2.to(args["device"])
            #     y2_len = y2_len.to(args["device"])

            # Noise augmentation is faster on GPU
            if args["whiteNoiseSD"] > 0:
                X += torch.randn(X.shape, device=args["device"]) * args["whiteNoiseSD"]

            if args["constantOffsetSD"] > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=args["device"])
                    * args["constantOffsetSD"]
                )

            adjustedLens = (
                model.module.compute_length(X_len)
                if is_distributed and hasattr(model, "module")
                else model.compute_length(X_len)
            )
            
            # Compute prediction error
            rep_for_future = None
            fut_pred_for_loss = None

            # ==========
            # MULTI-HEAD CASE (nClasses_2)
            # ==========
            if have_second:
                if use_future_pred and future_pred_weight > 0.0:
                    # Assume model supports return_rep=True here
                    pred, pred2, rep, fut_pred = model.forward(
                        X, X_len, dayIdx, return_rep=True
                    )
                    rep_for_future = rep
                    fut_pred_for_loss = fut_pred
                else:
                    # No reps used for contrastive in multi-head case
                    pred, pred2 = model.forward(X, X_len, dayIdx)
                    rep = None
                    rep_for_future = None
                    fut_pred_for_loss = None

                loss1 = forward_ctc(pred, adjustedLens, y,  y_len)
                loss2 = forward_ctc(pred2, adjustedLens, y2, y2_len)
                loss = loss1 + loss2
                train_ctc_loss.append(loss.cpu().detach().numpy())

                # For now, do NOT use multi-head path for contrastive (keeps it simple).
                logits_for_contrast = None
                rep_for_contrast = None

            # ==========
            # SINGLE-HEAD CASE
            # ==========
            else:
                if is_deep_ctc_model:
                    # -----------------------
                    # DEEP CTC MODEL
                    # -----------------------
                    if args.get('consistency', False):
                        # ===== CR-CTC consistency on main logits =====
                        from neural_decoder.augmentations import apply_specaugment_two_views
                        X1, X2 = apply_specaugment_two_views(X, X_len, args)

                        # Optional debug for first batch
                        if batch_idx == 0 and epoch == args["start_epoch"]:
                            x_diff = (X1 - X2).abs().sum().item()
                            print(f"🔍 Debug: X1 vs X2 difference (first batch): {x_diff:.6f}")
                            if x_diff < 1e-6:
                                print("⚠️  WARNING: X1 and X2 are nearly identical! Check augmentation parameters.")

                        # Two forward passes with deep CTC - ALWAYS get reps
                        if use_future_pred and future_pred_weight > 0.0:
                            if use_ntp and lambda_ntp > 0.0 and model_has_ntp:
                                logits_main1, deep_logits1, rep1, fut_pred1, ntp_logits1 = model(
                                    X1, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                                logits_main2, deep_logits2, rep2, fut_pred2, ntp_logits2 = model(
                                    X2, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                                ntp_logits = ntp_logits1
                            else:
                                logits_main1, deep_logits1, rep1, fut_pred1 = model(
                                    X1, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                logits_main2, deep_logits2, rep2, fut_pred2 = model(
                                    X2, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                ntp_logits = None
                            rep_for_future = rep1
                            fut_pred_for_loss = fut_pred1
                        else:
                            if use_ntp and lambda_ntp > 0.0 and model_has_ntp:
                                logits_main1, deep_logits1, rep1, _, ntp_logits1 = model(
                                    X1, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                                logits_main2, deep_logits2, rep2, _, ntp_logits2 = model(
                                    X2, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                                ntp_logits = ntp_logits1
                            else:
                                logits_main1, deep_logits1, rep1, _ = model(
                                    X1, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                logits_main2, deep_logits2, rep2, _ = model(
                                    X2, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                ntp_logits = None
                            rep_for_future = None
                            fut_pred_for_loss = None

                        # CR-CTC on main logits
                        encoder_out = torch.cat([logits_main1, logits_main2], dim=0)  # (2N, T, C)
                        encoder_out_lens = adjustedLens.repeat(2)                     # (2N,)
                        targets = torch.cat([y, y], dim=0)                            # (2 * sum(y_len))
                        target_lengths = y_len.repeat(2)                              # (2N,)

                        ctc_loss, kl_loss = forward_cr_ctc(
                            encoder_out,
                            encoder_out_lens,
                            targets,
                            target_lengths,
                        )

                        ctc_loss = 0.5 * ctc_loss
                        kl_loss = 0.5 * kl_loss

                        if batch_idx < 3 and epoch == args["start_epoch"]:
                            print(f"🔍 Debug batch {batch_idx}: KL loss = {kl_loss.item():.6f}")

                        main_loss = ctc_loss + args.get('consistency_scalar', 0.2) * kl_loss
                        train_kl_loss.append(kl_loss.detach().cpu().numpy())

                        # Deep CTC auxiliary losses (on first view)
                        deep_losses = []
                        for dl in deep_logits1:
                            deep_losses.append(forward_ctc(dl, adjustedLens, y, y_len))
                        if len(deep_losses) > 0:
                            deep_loss = sum(deep_losses) / len(deep_losses)
                            train_deep_loss_list.append(deep_loss.detach().cpu().numpy())
                        else:
                            deep_loss = logits_main1.new_tensor(0.0)

                        loss = main_loss + deep_ctc_weight * deep_loss
                        train_ctc_loss.append(main_loss.cpu().detach().numpy())
                        train_deep_loss_list.append(deep_loss.detach().cpu().numpy())
                        
                        # Add NTP loss if enabled (for consistency case)
                        if use_ntp and lambda_ntp > 0.0 and model_has_ntp and ntp_logits is not None:
                            # Generate NTP targets
                            B = ntp_logits.shape[0]
                            P_ntp = ntp_logits.shape[1]  # P-1 patches
                            ntp_targets = torch.full((B, P_ntp), fill_value=-100, dtype=torch.long, device=y.device)
                            
                            for b in range(B):
                                T_ntp = P_ntp
                                U_b = y_len[b].item()
                                if T_ntp > 0 and U_b > 0:
                                    gt_labels = y[b, :U_b]  # [U_b]
                                    for t in range(min(T_ntp, len(gt_labels) - 1)):
                                        label_idx = int(t * (len(gt_labels) - 1) / max(1, T_ntp - 1))
                                        ntp_targets[b, t] = gt_labels[label_idx + 1].item()  # Next token
                            
                            ntp_loss = F.cross_entropy(
                                ntp_logits.reshape(-1, ntp_logits.shape[-1]),
                                ntp_targets.reshape(-1),
                                ignore_index=-100,
                            )
                            loss = loss + lambda_ntp * ntp_loss
                            train_ntp_losses.append(ntp_loss.detach().cpu().numpy())
                        else:
                            train_ntp_losses.append(0.0)

                        # Store rep/logits for phone contrastive (use view 1)
                        if use_phone_contrastive and phone_contrastive_weight > 0.0:
                            logits_for_contrast = logits_main1
                            rep_for_contrast = rep1

                    else:
                        # Deep CTC: get main + intermediate logits + reps ONCE
                        if use_future_pred and future_pred_weight > 0.0:
                            if use_ntp and lambda_ntp > 0.0 and model_has_ntp:
                                logits_main, deep_logits, rep, fut_pred, ntp_logits = model(
                                    X, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                            else:
                                logits_main, deep_logits, rep, fut_pred = model(
                                    X, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                ntp_logits = None
                            rep_for_future = rep
                            fut_pred_for_loss = fut_pred
                        else:
                            if use_ntp and lambda_ntp > 0.0 and model_has_ntp:
                                logits_main, deep_logits, rep, _, ntp_logits = model(
                                    X, X_len, dayIdx, return_deep=True, return_rep=True, return_ntp=True
                                )
                            else:
                                logits_main, deep_logits, rep, _ = model(
                                    X, X_len, dayIdx, return_deep=True, return_rep=True
                                )
                                ntp_logits = None
                            rep_for_future = None
                            fut_pred_for_loss = None
                        
                        # Store rep/logits for phone contrastive loss (if enabled)
                        if use_phone_contrastive and phone_contrastive_weight > 0.0:
                            logits_for_contrast = logits_main
                            rep_for_contrast = rep  # encoder representation
                        
                        # Compute CTC loss (or combined CTC+NTP loss if NTP enabled)
                        if use_ntp and lambda_ntp > 0.0 and model_has_ntp and ntp_logits is not None:
                            # Generate NTP targets: distribute ground truth labels across patches and shift by 1
                            B = ntp_logits.shape[0]
                            P_ntp = ntp_logits.shape[1]  # ntp_logits: [B, P-1, C]
                            ntp_targets = torch.full((B, P_ntp), fill_value=-100, dtype=torch.long, device=y.device)
                            
                            for b in range(B):
                                T_ntp = P_ntp
                                U_b = y_len[b].item()
                                if T_ntp > 0 and U_b > 0:
                                    # Distribute ground truth labels across patches
                                    gt_labels = y[b, :U_b]  # [U_b]
                                    # Map each patch position to a label index
                                    for t in range(min(T_ntp, len(gt_labels) - 1)):
                                        label_idx = int(t * (len(gt_labels) - 1) / max(1, T_ntp - 1))
                                        ntp_targets[b, t] = gt_labels[label_idx + 1].item()  # Next token
                            
                            # Use combined CTC+NTP loss
                            main_loss, ctc_loss_only, ntp_loss = forward_ctc_ntp(
                                encoder_out=logits_main,
                                encoder_out_lens=adjustedLens,
                                targets=y,
                                target_lengths=y_len,
                                ntp_logits=ntp_logits,
                                ntp_targets=ntp_targets,
                                lambda_ntp=lambda_ntp,
                                ntp_ignore_index=-100,
                            )
                            train_ntp_losses.append(ntp_loss.detach().cpu().numpy())
                        else:
                            main_loss = forward_ctc(logits_main, adjustedLens, y, y_len)
                            if not (use_ntp and lambda_ntp > 0.0):
                                train_ntp_losses.append(0.0)

                        deep_losses = []
                        for dl in deep_logits:
                            deep_losses.append(forward_ctc(dl, adjustedLens, y, y_len))
                        if len(deep_losses) > 0:
                            deep_loss = sum(deep_losses) / len(deep_losses)
                            loss = main_loss + deep_ctc_weight * deep_loss
                            train_ctc_loss.append(main_loss.cpu().detach().numpy())
                            train_deep_loss_list.append(deep_loss.detach().cpu().numpy())
                        else:
                            loss = main_loss
                            
                else:
                    # -----------------------
                    # STANDARD (non-deep) MODEL
                    # -----------------------
                    if is_hierarchical_model:
                        # Hierarchical CTC
                        if use_future_pred and future_pred_weight > 0.0:
                            logits_phone, logits_broad, rep, fut_pred = model.forward(
                                X, X_len, dayIdx, return_rep=True
                            )
                            rep_for_future = rep
                            fut_pred_for_loss = fut_pred
                        elif use_phone_contrastive and phone_contrastive_weight > 0.0:
                            # Need reps for contrastive even without future pred
                            logits_phone, logits_broad, rep, _ = model.forward(
                                X, X_len, dayIdx, return_rep=True
                            )
                            rep_for_future = None
                            fut_pred_for_loss = None
                        else:
                            logits_phone, logits_broad = model.forward(
                                X, X_len, dayIdx
                            )
                            rep = None
                            rep_for_future = None
                            fut_pred_for_loss = None

                        hier_aux_weight = args.get("hier_aux_weight", 0.3)
                        loss, loss_phone, loss_broad = compute_hierarchical_ctc_loss(
                            logits_phone,
                            logits_broad,
                            adjustedLens,
                            y,
                            y_len,
                            blank_index=0,
                            hier_aux_weight=hier_aux_weight,
                        )
                        train_hier_phone_losses.append(loss_phone.cpu().detach().numpy())
                        train_hier_broad_losses.append(loss_broad.cpu().detach().numpy())
                        train_ctc_loss.append(loss_phone.cpu().detach().numpy())
                        pred = logits_phone

                        if use_phone_contrastive and phone_contrastive_weight > 0.0:
                            logits_for_contrast = logits_phone
                            rep_for_contrast = rep

                    else:
                        # Non-hierarchical, non-deep
                        if args.get('consistency', False):
                            from neural_decoder.augmentations import apply_specaugment_two_views
                            X1, X2 = apply_specaugment_two_views(X, X_len, args)
                            
                            # Debug: Check if X1 and X2 are different
                            if batch_idx == 0 and epoch == args["start_epoch"]:
                                x_diff = (X1 - X2).abs().sum().item()
                                print(f"🔍 Debug: X1 vs X2 difference (first batch): {x_diff:.6f}")
                                if x_diff < 1e-6:
                                    print("⚠️  WARNING: X1 and X2 are nearly identical! Check augmentation parameters.")
                            
                            # Two forward passes – same model, different view
                            if use_future_pred and future_pred_weight > 0.0:
                                pred1, rep1, fut_pred1 = model(
                                    X1, X_len, dayIdx, return_rep=True
                                )
                                pred2, rep2, fut_pred2 = model(
                                    X2, X_len, dayIdx, return_rep=True
                                )
                                rep_for_future = rep1
                                fut_pred_for_loss = fut_pred1
                            elif use_phone_contrastive and phone_contrastive_weight > 0.0:
                                pred1, rep1, _ = model(
                                    X1, X_len, dayIdx, return_rep=True
                                )
                                pred2, rep2, _ = model(
                                    X2, X_len, dayIdx, return_rep=True
                                )
                                rep_for_future = None
                                fut_pred_for_loss = None
                            else:
                                pred1 = model(X1, X_len, dayIdx)   # (N, T, C)
                                pred2 = model(X2, X_len, dayIdx)   # (N, T, C)
                                rep_for_future = None
                                fut_pred_for_loss = None

                            encoder_out = torch.cat([pred1, pred2], dim=0)        # (2N, T, C)
                            encoder_out_lens = adjustedLens.repeat(2)             # (2N,)
                            targets = torch.cat([y, y], dim=0)                    # (2 * sum(y_len))
                            target_lengths = y_len.repeat(2)                      # (2N,)

                            ctc_loss, kl_loss = forward_cr_ctc(
                                encoder_out,
                                encoder_out_lens,
                                targets,
                                target_lengths,
                            )

                            ctc_loss = ctc_loss * 0.5
                            kl_loss = kl_loss * 0.5

                            # Debug: Print KL loss for first few batches
                            if batch_idx < 3 and epoch == args["start_epoch"]:
                                print(f"🔍 Debug batch {batch_idx}: KL loss = {kl_loss.item():.6f}")

                            loss = ctc_loss + args.get('consistency_scalar', 0.2) * kl_loss
                            train_kl_loss.append(kl_loss.detach().cpu().numpy())
                            train_ctc_loss.append(ctc_loss.cpu().detach().numpy())

                            if use_phone_contrastive and phone_contrastive_weight > 0.0:
                                logits_for_contrast = pred1
                                rep_for_contrast = rep1

                        else:
                            if use_future_pred and future_pred_weight > 0.0:
                                pred, rep, fut_pred = model.forward(
                                    X, X_len, dayIdx, return_rep=True
                                )
                                rep_for_future = rep
                                fut_pred_for_loss = fut_pred
                            elif use_phone_contrastive and phone_contrastive_weight > 0.0:
                                pred, rep, _ = model.forward(
                                    X, X_len, dayIdx, return_rep=True
                                )
                                rep_for_future = None
                                fut_pred_for_loss = None
                            else:
                                pred = model.forward(X, X_len, dayIdx)
                                rep = None
                                rep_for_future = None
                                fut_pred_for_loss = None

                            loss = forward_ctc(pred, adjustedLens, y, y_len)
                            train_ctc_loss.append(loss.cpu().detach().numpy())

                            if use_phone_contrastive and phone_contrastive_weight > 0.0:
                                logits_for_contrast = pred
                                rep_for_contrast = rep

            # ==========
            # CPC on phone representations (future prediction loss)
            # ==========
            if (
                use_future_pred
                and future_predictor is not None
                and future_pred_weight > 0.0
                and rep_for_future is not None
                and is_deep_ctc_model  # Currently only supported for deep CTC
            ):
                fut_loss = future_prediction_loss(
                    rep_for_future,      # (B, T, D) - encoder representation
                    adjustedLens,        # (B,)
                    future_predictor,
                    steps=future_pred_steps,
                    loss_type="cosine",
                )
                loss = loss + future_pred_weight * fut_loss
                train_future_pred_losses.append(fut_loss.detach().cpu().numpy())

            # ==========
            # Phone contrastive loss (InfoNCE) across trials
            # ==========
            if (
                use_phone_contrastive
                and phone_contrastive_weight > 0.0
                and rep_for_contrast is not None
                and logits_for_contrast is not None
                and epoch >= phone_contrastive_start_epoch
            ):
                with torch.no_grad():
                    phone_ids, valid_mask_base = ctc_run_alignment_phone_ids(
                        logits=logits_for_contrast,  # [B, T, C]
                        targets=y,                   # [B, U_max]
                        input_lengths=adjustedLens,  # [B]
                        target_lengths=y_len,        # [B]
                        blank=0,
                    )

                    probs = torch.softmax(logits_for_contrast, dim=-1)  # [B, T, C]
                    max_probs, _ = probs.max(dim=-1)                    # [B, T]

                    conf_thresh = args.get("phone_contrastive_conf_thresh", 0.7)
                    conf_mask = max_probs > conf_thresh                 # [B, T]

                    # final valid mask: within length + high confidence;
                    # phone_contrastive_loss / cross_trial_phone_contrastive_loss
                    # will also enforce (phone_ids >= 0)
                    valid_mask = valid_mask_base & conf_mask
                if args.get("use_cross_trial_phone_contrastive_loss", True):
                    phon_contrast_loss = cross_trial_phone_contrastive_loss(
                        reps=rep_for_contrast,          # [B, T, D] encoder reps
                        phone_ids=phone_ids,            # [B, T]
                        valid_mask=valid_mask,          # [B, T]
                        temperature=phone_contrastive_temperature,
                        max_frames=phone_contrastive_max_samples,
                    )
                else:
                    phon_contrast_loss = phone_contrastive_loss(
                        reps=rep_for_contrast,          # [B, T, D] encoder reps
                        phone_ids=phone_ids,            # [B, T]
                        valid_mask=valid_mask,          # [B, T]
                        temperature=phone_contrastive_temperature,
                        max_frames=phone_contrastive_max_samples,
                    )

                loss = loss + phone_contrastive_weight * phon_contrast_loss
                train_phone_contrastive_losses.append(
                    phon_contrast_loss.detach().cpu().numpy()
                )

            train_loss.append(loss.cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
    
        with torch.no_grad():
    
            avgTrainLoss = np.mean(train_loss) if len(train_loss) > 0 else 0.0
            avgTrainKLLoss = np.mean(train_kl_loss) if len(train_kl_loss) > 0 else 0.0
            avgTrainDeepLoss = np.mean(train_deep_loss_list) if len(train_deep_loss_list) > 0 else 0.0
            avgTrainHierPhoneLoss = np.mean(train_hier_phone_losses) if len(train_hier_phone_losses) > 0 else 0.0
            avgTrainHierBroadLoss = np.mean(train_hier_broad_losses) if len(train_hier_broad_losses) > 0 else 0.0
            avgTrainFuturePredLoss = np.mean(train_future_pred_losses) if len(train_future_pred_losses) > 0 else 0.0
            avgTrainPhoneContrastiveLoss = np.mean(train_phone_contrastive_losses) if len(train_phone_contrastive_losses) > 0 else 0.0
            avgTrainNTPLoss = np.mean(train_ntp_losses) if len(train_ntp_losses) > 0 else 0.0

            # Synchronize loss values across all DDP processes
            if is_distributed:
                # Convert to tensor, reduce across all processes, then convert back
                avgTrainLoss_tensor = torch.tensor(avgTrainLoss, device=args["device"])
                dist.all_reduce(avgTrainLoss_tensor, op=dist.ReduceOp.SUM)
                avgTrainLoss = (avgTrainLoss_tensor.item() / world_size)
                
                if args.get('consistency', False) and len(train_kl_loss) > 0:
                    avgTrainKLLoss_tensor = torch.tensor(avgTrainKLLoss, device=args["device"])
                    dist.all_reduce(avgTrainKLLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainKLLoss = (avgTrainKLLoss_tensor.item() / world_size)
                
                if is_deep_ctc_model and len(train_deep_loss_list) > 0:
                    avgTrainDeepLoss_tensor = torch.tensor(avgTrainDeepLoss, device=args["device"])
                    dist.all_reduce(avgTrainDeepLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainDeepLoss = (avgTrainDeepLoss_tensor.item() / world_size)

                if is_hierarchical_model and len(train_hier_phone_losses) > 0:
                    avgTrainHierPhoneLoss_tensor = torch.tensor(avgTrainHierPhoneLoss, device=args["device"])
                    avgTrainHierBroadLoss_tensor = torch.tensor(avgTrainHierBroadLoss, device=args["device"])
                    dist.all_reduce(avgTrainHierPhoneLoss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avgTrainHierBroadLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainHierPhoneLoss = avgTrainHierPhoneLoss_tensor.item() / world_size
                    avgTrainHierBroadLoss = avgTrainHierBroadLoss_tensor.item() / world_size

                if use_future_pred and len(train_future_pred_losses) > 0:
                    avgTrainFuturePredLoss_tensor = torch.tensor(avgTrainFuturePredLoss, device=args["device"])
                    dist.all_reduce(avgTrainFuturePredLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainFuturePredLoss = avgTrainFuturePredLoss_tensor.item() / world_size

                if use_phone_contrastive and len(train_phone_contrastive_losses) > 0:
                    avgTrainPhoneContrastiveLoss_tensor = torch.tensor(avgTrainPhoneContrastiveLoss, device=args["device"])
                    dist.all_reduce(avgTrainPhoneContrastiveLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainPhoneContrastiveLoss = avgTrainPhoneContrastiveLoss_tensor.item() / world_size
                
                if use_ntp and len(train_ntp_losses) > 0:
                    avgTrainNTPLoss_tensor = torch.tensor(avgTrainNTPLoss, device=args["device"])
                    dist.all_reduce(avgTrainNTPLoss_tensor, op=dist.ReduceOp.SUM)
                    avgTrainNTPLoss = avgTrainNTPLoss_tensor.item() / world_size

            # Compute train CER - only on rank 0 using full dataset (no sampler) to mimic single GPU
            model.eval()
            train_total_edit_distance = 0
            train_total_seq_length = 0
            
            if not is_distributed or rank == 0:
                # Sample a few batches from full training data (no DistributedSampler)
                for train_batch_idx, train_batch in enumerate(trainMetricsLoader):
                    if train_batch_idx >= 10:  # Sample first 10 batches for train CER (faster)
                        break

                    X_train, X_len_train, y_train, y_len_train, dayIdx_train, y2_train, y2_len_train = unpack_batch_5or7(train_batch)
                    X_train = X_train.to(args["device"])
                    y_train = y_train.to(args["device"])
                    X_len_train = X_len_train.to(args["device"])
                    y_len_train = y_len_train.to(args["device"])
                    dayIdx_train = dayIdx_train.to(args["device"])
                    
                    adjustedLens_train = (
                        model.module.compute_length(X_len_train)
                        if is_distributed and hasattr(model, "module")
                        else model.compute_length(X_len_train)
                    )

                    if is_hierarchical_model:
                        pred_train, _ = model.forward(X_train, X_len_train, dayIdx_train)
                    else:
                        pred_train = model.forward(X_train, X_len_train, dayIdx_train)
                    
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
                
                train_cer = (
                    train_total_edit_distance / train_total_seq_length
                    if train_total_seq_length > 0 else float('nan')
                )
            else:
                train_cer = float('nan')
            
            # Evaluate on test set
            allLoss = []
            total_edit_distance = 0
            total_seq_length = 0
            have_second = args.get('nClasses_2') is not None
            if have_second:
                total_edit_distance2 = 0
                total_seq_length2 = 0

            model.eval()

            if (not is_distributed) or (rank == 0):
                for batch in testLoader:
                    # Check batch length to handle datasets with or without text2
                    # Batch can have 5 items (no text2) or 7 items (with text2)
                    X, X_len, y, y_len, testDayIdx, y2, y2_len = unpack_batch_5or7(batch)

                    if args['maxDay'] is not None:
                        testDayIdx.fill_(args['maxDay'])

                    X = X.to(args["device"])
                    y = y.to(args["device"])
                    X_len = X_len.to(args["device"])
                    y_len = y_len.to(args["device"])
                    testDayIdx = testDayIdx.to(args["device"])
                    if have_second and y2 is not None:
                        y2 = y2.to(args["device"])
                        y2_len = y2_len.to(args["device"])

                    adjustedLens = (
                        model.module.compute_length(X_len)
                        if is_distributed and hasattr(model, "module")
                        else model.compute_length(X_len)
                    )

                    if have_second:
                        pred, pred2 = model.forward(X, X_len, testDayIdx)
                        loss1 = forward_ctc(pred,  adjustedLens, y,  y_len)
                        loss2 = forward_ctc(pred2, adjustedLens, y2, y2_len)
                        loss = loss1 + loss2
                    else:
                        if is_hierarchical_model:
                            logits_phone, logits_broad = model.forward(
                                X, X_len, testDayIdx
                            )
                            hier_aux_weight = args.get("hier_aux_weight", 0.3)
                            loss, _, _ = compute_hierarchical_ctc_loss(
                                logits_phone,
                                logits_broad,
                                adjustedLens,
                                y,
                                y_len,
                                blank_index=0,
                                hier_aux_weight=hier_aux_weight,
                            )
                            pred = logits_phone
                        else:
                            pred = model.forward(X, X_len, testDayIdx)
                            loss = forward_ctc(pred, adjustedLens, y, y_len)

                    allLoss.append(loss.item())

                    # CER head 1
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
                        matcher = SequenceMatcher(a=trueSeq.tolist(),
                                                  b=decodedSeq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                    # CER head 2
                    if have_second:
                        for iterIdx in range(pred2.shape[0]):
                            decodedSeq2 = torch.argmax(
                                pred2[iterIdx, 0:adjustedLens[iterIdx], :], dim=-1
                            )
                            decodedSeq2 = torch.unique_consecutive(decodedSeq2, dim=-1)
                            decodedSeq2 = decodedSeq2.cpu().detach().numpy()
                            decodedSeq2 = np.array([i for i in decodedSeq2 if i != 0])

                            trueSeq2 = np.array(
                                y2[iterIdx][0:y2_len[iterIdx]].cpu().detach()
                            )
                            matcher = SequenceMatcher(a=trueSeq2.tolist(),
                                                      b=decodedSeq2.tolist())
                            total_edit_distance2 += matcher.distance()
                            total_seq_length2 += len(trueSeq2)

                avgDayLoss = np.mean(allLoss) if allLoss else 0.0
                cer = (total_edit_distance / total_seq_length
                       if total_seq_length > 0 else float('nan'))
                if have_second:
                    cer2 = (total_edit_distance2 / total_seq_length2
                            if total_seq_length2 > 0 else float('nan'))
            else:
                # other ranks don't eval / log
                avgDayLoss = float('nan')
                cer = float('nan')
                cer2 = float('nan') if have_second else None

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            endTime = time.time()
            elapsed_time = endTime - epoch_start_time
            time_per_batch = elapsed_time / num_batches if num_batches > 0 else 0.0
            
            # Only print on rank 0 to avoid duplicate output in distributed training
            if not is_distributed or rank == 0:
                msg = (
                    f"Epoch {epoch}, train ctc: {avgTrainLoss:>7f}, train cer: {train_cer:>7f}, "
                    f"eval ctc: {avgDayLoss:>7f}, eval cer: {cer:>7f}"
                )
                if have_second:
                    msg += f", eval cer2: {cer2:>7f}"
                if is_deep_ctc_model and len(train_deep_loss_list) > 0:
                    msg += f", train deep-ctc: {avgTrainDeepLoss:>7f}"
                if is_hierarchical_model and len(train_hier_phone_losses) > 0:
                    msg += f", hier phone: {avgTrainHierPhoneLoss:>7f}, hier broad: {avgTrainHierBroadLoss:>7f}"
                if use_future_pred and len(train_future_pred_losses) > 0:
                    msg += f", future_pred: {avgTrainFuturePredLoss:>7f}"
                if use_phone_contrastive and len(train_phone_contrastive_losses) > 0:
                    msg += f", phone_contrastive: {avgTrainPhoneContrastiveLoss:>7f}"
                if use_ntp and len(train_ntp_losses) > 0:
                    msg += f", ntp: {avgTrainNTPLoss:>7f}"
                msg += f", lr: {current_lr:.6f}, time/batch: {time_per_batch:>7.3f}"
                print(msg)
            
            # Log the metrics to wandb with requested format
            log_dict = {
                "train/ctc_loss": avgTrainLoss,
                "train/cer": train_cer,
                "eval/ctc_loss": avgDayLoss,
                "eval/cer": cer,
                "train/learning_rate": current_lr,
                "epoch": epoch + 1,
            }
            if have_second:
                log_dict["eval/cer2"] = cer2   # Secondary CER if applicable
                
            if args.get('consistency', False) and len(train_kl_loss) > 0:
                log_dict['train/kl-loss'] = avgTrainKLLoss
            if is_deep_ctc_model and len(train_deep_loss_list) > 0:
                log_dict["train/deep_ctc_loss"] = avgTrainDeepLoss
            if is_hierarchical_model and len(train_hier_phone_losses) > 0:
                log_dict["train/hier_phone_loss"] = avgTrainHierPhoneLoss
                log_dict["train/hier_broad_loss"] = avgTrainHierBroadLoss
            if use_future_pred and len(train_future_pred_losses) > 0:
                log_dict["train/future_pred_loss"] = avgTrainFuturePredLoss
            if use_phone_contrastive and len(train_phone_contrastive_losses) > 0:
                log_dict["train/phone_contrastive_loss"] = avgTrainPhoneContrastiveLoss
            if use_ntp and len(train_ntp_losses) > 0:
                log_dict["train/ntp_loss"] = avgTrainNTPLoss

            # Only log to wandb on rank 0
            if wandb.run is not None and (not is_distributed or rank == 0):
                wandb.log(log_dict)

        # Only save checkpoints on rank 0
        if (not is_distributed or rank == 0):
            if len(testCER) > 0 and cer < np.min(testCER):
                # Handle DDP model state dict saving
                model_state = (
                    model.module.state_dict()
                    if is_distributed and hasattr(model, 'module')
                    else model.state_dict()
                )
                torch.save(model_state, args["outputDir"] + "/modelWeights")
                torch.save(optimizer.state_dict(), args["outputDir"] + "/optimizer")
                torch.save(scheduler.state_dict(), args['outputDir'] + '/scheduler')
                
            if len(testLoss) > 0 and avgDayLoss < np.min(testLoss):
                model_state = (
                    model.module.state_dict()
                    if is_distributed and hasattr(model, 'module')
                    else model.state_dict()
                )
                torch.save(model_state, args["outputDir"] + "/modelWeights_ctc")
                
            if have_second:
                if len(testCER2) > 0 and cer2 < np.min(testCER2):
                    model_state = (
                        model.module.state_dict()
                        if is_distributed and hasattr(model, 'module')
                        else model.state_dict()
                    )
                    torch.save(model_state, args["outputDir"] + "/modelWeights2")
                
        testLoss.append(avgDayLoss)
        testCER.append(cer)
        if have_second:
            testCER2.append(cer2)

        tStats = {}
        tStats["testLoss"] = np.array(testLoss)
        tStats["testCER"] = np.array(testCER)

        with open(args["outputDir"] + "/trainingStats", "wb") as file:
            pickle.dump(tStats, file)
            
        scheduler.step()
                    
    # Only finish wandb on rank 0 (where it was initialized)
    if (not is_distributed or rank == 0) and wandb.run is not None:
        wandb.finish()
    return
