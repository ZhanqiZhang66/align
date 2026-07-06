import os
import torch
import torch.distributed as dist
import numpy as np
import argparse
import yaml
from pathlib import Path
# torchrun --nproc_per_node=4 scripts/train_transformer.py

# CRITICAL: Set WandB mode based on rank BEFORE importing wandb
# torchrun sets RANK as an environment variable before spawning processes
rank = int(os.environ.get('RANK', 0))
if rank == 0:
    os.environ['WANDB_MODE'] = "online"  # Only rank 0 logs to WandB
else:
    os.environ['WANDB_MODE'] = "disabled"  # Disable WandB on all other ranks

# Set WandB API key (needed for all ranks, but only rank 0 will use it)
os.environ['WANDB_API_KEY'] = "57ee39f7bd565c158fe30f4e0466594e03bca347"

# Initialize DDP if using torchrun (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:{os.environ.get('MASTER_PORT', '12355')}",
            rank=rank,
            world_size=world_size
        )
        if rank == 0:
            print(f"✅ DDP initialized: world_size={world_size}, rank={rank}, local_rank={local_rank}")

from neural_decoder.neural_decoder_trainer_align import trainModelDANN
from neural_decoder.bit_align import BiT_Phoneme

# === CONFIGURATION ===
BASE_PATHS = {
    'obi': '/victoriapvc/data/',
    'leia': '/victoriapvc/data/'
}

DATA_PATHS = {
    'obi': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc'),
    'obi_log': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both'),
    'obi_log_char': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_char'),
    'obi_log_char_phoneme': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_char_phoneme'),
    'obi_log_held_out': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days'),
    'obi_log_held_out_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_1'),
    'obi_log_held_out_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_2'), 
    'obi_log_held_out_interleaved': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_interleaved'),
    'obi_log_big_0': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_big_0'), 
    'obi_log_big_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_big_1'),
    'obi_log_big_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_big_2'),
    'obi_log_12_0_4': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_12_0_4'),
    'obi_log_12_0_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_12_0_1'),
    'obi_log_16_0_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_16_0_1'),
    'obi_log_19_0_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_19_0_1'),
    'obi_log_18_1_1': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_18_1_1'),
    # add 15_1_4, 16_0_4, 16_1_3, 17_0_3, 17_1_2, 18_0_2, 19_0_1, 18_1_1
    'obi_log_15_1_4': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_15_1_4'),
    'obi_log_16_0_4': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_16_0_4'),
    'obi_log_16_1_3': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_16_1_3'),
    'obi_log_17_0_3': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_17_0_3'),
    'obi_log_17_1_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_17_1_2'),
    'obi_log_18_0_2': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_18_0_2'),
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_char': os.path.join(BASE_PATHS['leia'], 'data_log_both_char'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days'), 
    'leia_log_held_out_1': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_1'), 
    'leia_log_held_out_2': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_2'),
    'leia_log_char_phoneme': os.path.join(BASE_PATHS['leia'], 'data_log_both_char_and_phoneme')
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Transformer model (MDAN GRL real)')
parser.add_argument('--config', type=str, 
                    default='scripts/nrp/train_transformer_cross_session_align_config.yaml',
                    help='Path to YAML configuration file')
args_cli = parser.parse_args()

# Load YAML configuration
config_path = Path(args_cli.config)
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration sections
paths_config = config.get('paths', {})
training_config = config.get('training', {})
model_config = config.get('model', {})
data_config = config.get('data', {})
masking_config = config.get('masking', {})
optimization_config = config.get('optimization', {})
dann_config = config.get('dann', {})
other_config = config.get('other', {})

# Get values from config
seed_list = training_config.get('seed_list', [0])
SERVER = paths_config.get('server', 'obi')
DATA_PATH_KEY = paths_config.get('data_path_key', f"{SERVER}_log")
model_name_base = training_config.get('model_name_base', '')
CHECKPOINT_DIR = training_config.get('checkpoint_dir', '')
wandb_id = training_config.get('wandb_id', '')
RESUME_FROM_EPOCH = training_config.get('resume_from_epoch', 0)

# === MAIN LOOP ===
for seed in seed_list:
    
    model_name = f"{model_name_base}_seed_{seed}"
    output_dir = os.path.join(BASE_PATHS[SERVER], 'outputs', model_name)
    dataset_path = DATA_PATHS[DATA_PATH_KEY]

    # Get source_days and target_days from dann_config, with defaults based on DATA_PATH_KEY
    if DATA_PATH_KEY == "obi_log_held_out":
        default_source_days = list(range(0, 15))
        default_target_days = list(range(15, 20))
    elif DATA_PATH_KEY == "obi_log_big_0":
        default_source_days = list(range(0, 12))
        default_target_days = list[int](range(12, 20))
    elif DATA_PATH_KEY == "obi_log_big_1":
        default_source_days = list(range(0, 10))
        default_target_days = list(range(12, 20))
    elif DATA_PATH_KEY == "obi_log_big_2":
        default_source_days = list(range(0, 12))
        default_target_days = list(range(16, 20))
    # add 15_1_4, 16_0_4, 16_1_3, 17_0_3, 17_1_2, 18_0_2, 19_0_1, 18_1_1
    # first is source days, third is target days and second is gap days
    elif DATA_PATH_KEY == "obi_log_15_1_4":
        default_source_days = list(range(0, 15))
        default_target_days = list(range(16, 20))
    elif DATA_PATH_KEY == "obi_log_16_0_4":
        default_source_days = list(range(0, 16))
        default_target_days = list(range(16, 20))
    elif DATA_PATH_KEY == "obi_log_16_1_3":
        default_source_days = list(range(0, 16))
        default_target_days = list(range(17, 20))
    elif DATA_PATH_KEY == "obi_log_17_0_3":
        default_source_days = list(range(0, 17))
        default_target_days = list(range(17, 20))
    elif DATA_PATH_KEY == "obi_log_17_1_2":
        default_source_days = list(range(0, 17))
        default_target_days = list(range(18, 20))
    elif DATA_PATH_KEY == "obi_log_18_0_2":
        default_source_days = list(range(0, 18))
        default_target_days = list(range(18, 20))
    elif DATA_PATH_KEY == "obi_log_19_0_1":
        default_source_days = list(range(0, 19))
        default_target_days = list(range(19, 20))
    elif DATA_PATH_KEY == "obi_log_18_1_1":
        default_source_days = list(range(0, 18))
        default_target_days = list(range(19, 20))
    elif DATA_PATH_KEY == "obi_log_12_0_4":
        default_source_days = list(range(0, 12))
        default_target_days = list(range(12, 16))
    elif DATA_PATH_KEY == "obi_log_12_0_1":
        default_source_days = list(range(0, 12))
        default_target_days = list(range(12, 13))
    elif DATA_PATH_KEY == "obi_log_16_0_1":
        default_source_days = list(range(0, 16))
        default_target_days = list(range(16, 17))
    elif DATA_PATH_KEY == "obi_log_12_8_3_no_test":
        default_source_days = list(range(0, 12))
        default_target_days = list(range(20, 23))
    elif DATA_PATH_KEY == "obi_log_12_2gap_4_2gap_3_test":
        default_source_days = list(range(0, 12))
        default_target_days = list(range(14, 18))
    else:
        default_source_days = list(range(0, 10))
        default_target_days = list(range(9, 20))
    
    source_days = dann_config.get("source_days", default_source_days)
    target_days = dann_config.get("target_days", default_target_days)
    
    if not isinstance(source_days, list):
        source_days = default_source_days
    if not isinstance(target_days, list):
        target_days = default_target_days
    
    if rank == 0:
        print(f"🔀 DANN Configuration:")
        print(f"   Source days: {source_days}")
        print(f"   Target days: {target_days}")
    
    patch_size = tuple(model_config.get('patch_size', [5, 256]))
    
    # Create config dictionary from YAML config
    args = {
        'seed': seed,
        'outputDir': output_dir,
        'datasetPath': dataset_path,
        'modelName': model_name,
        'maxDay': data_config.get('maxDay'),
        'restricted_days': data_config.get('restricted_days', []),
        'patch_size': patch_size,
        'dim': model_config.get('dim', 384),
        'depth': model_config.get('depth', 5),
        'heads': model_config.get('heads', 6),
        'mlp_dim_ratio': model_config.get('mlp_dim_ratio', 4),
        'dim_head': model_config.get('dim_head', 64),
        'T5_style_pos': model_config.get('T5_style_pos', True),
        'nClasses': model_config.get('nClasses', 40),
        'nClasses_2': model_config.get('nClasses_2'),
        'whiteNoiseSD': data_config.get('whiteNoiseSD', 0.2),
        'gaussianSmoothWidth': data_config.get('gaussianSmoothWidth', 2.0),
        'constantOffsetSD': data_config.get('constantOffsetSD', 0.05),
        'l2_decay': optimization_config.get('l2_decay', 1e-5),
        'input_dropout': model_config.get('input_dropout', 0.2),
        'dropout': model_config.get('dropout', 0.35),
        'AdamW': optimization_config.get('AdamW', True),
        'learning_scheduler': optimization_config.get('learning_scheduler', 'multistep'),
        'lrStart': optimization_config.get('lrStart', 0.001),
        'lrEnd': optimization_config.get('lrEnd', 0.001),
        'batchSize': optimization_config.get('batchSize', 64),
        'beta1': optimization_config.get('beta1', 0.90),
        'beta2': optimization_config.get('beta2', 0.999),
        'n_epochs': optimization_config.get('n_epochs', 250),
        'milestones': optimization_config.get('milestones', [150]),
        'gamma': optimization_config.get('gamma', 0.1),
        'extra_notes': other_config.get('extra_notes', ""),
        'device': f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if torch.cuda.is_available() else 'cpu',
        'load_pretrained_model': CHECKPOINT_DIR,
        'wandb_id': wandb_id,
        'start_epoch': RESUME_FROM_EPOCH,
        'ventral_6v_only': other_config.get('ventral_6v_only', False),
        'mask_token_zero': masking_config.get('mask_token_zero', False),
        'num_masks_channels': masking_config.get('num_masks_channels', 4),
        'max_mask_channels': masking_config.get('max_mask_channels', 4),
        'max_mask_pct': masking_config.get('max_mask_pct', 0.075), 
        'num_masks': masking_config.get('num_masks', 20),
        'dist_dict_path': masking_config.get('dist_dict_path', '/home3/skaasyap/willett/outputs/dist_dict.pt'), 
        'consistency': masking_config.get('consistency', False), 
        'consistency_scalar': masking_config.get('consistency_scalar', 0.2),
        'wandb_project': other_config.get('wandb_project', 'brain2text-debug'),
        'wandb_entity': other_config.get('wandb_entity', 'victoriazhang-projects'),
        'wandb_name': model_name_base,
        "use_dann": True,
        "dann_lambda": dann_config.get("dann_lambda", 0.001),
        "lambda_src": dann_config.get("lambda_src", 1.0),
        "lambda_tgt": dann_config.get("lambda_tgt", 0.05),
        "dann_warmup_epochs": dann_config.get("dann_warmup_epochs", 0),
        "dann_lr_multiplier": dann_config.get("dann_lr_multiplier", 1.0),
        "dann_weight_decay": dann_config.get("dann_weight_decay", None),
        "dann_hidden": dann_config.get("domain_hidden", 256),
        "domain_dropout": dann_config.get("domain_dropout", 0.1),
        "phase_disc_epochs": dann_config.get("phase_disc_epochs", 10),
        "phase_encoder_epochs": dann_config.get("phase_encoder_epochs", 1),
        "use_amp": dann_config.get("use_amp", True),
        "grad_clip": dann_config.get("grad_clip", 1.0),
        "grad_clip_disc": dann_config.get("grad_clip_disc", 1.0),
        "source_days": source_days,
        "target_days": target_days,
        "rep_layer_idx": dann_config.get("rep_layer_idx", 4),
        "dann_optimizer": dann_config.get("dann_optimizer", "AdamW"),
        "use_me_loss": dann_config.get("use_me_loss", True),
        "dann_alpha_type": dann_config.get("dann_alpha_type", "standard"),
        "use_target_loss": dann_config.get("use_target_loss", False),
        "linear_discriminator": dann_config.get("linear_discriminator", True),
        "dann_dropout_channel_prob": dann_config.get("dann_dropout_channel_prob", 0.0),
        "detach_discriminator_from_encoder": dann_config.get("detach_discriminator_from_encoder", False),
        "bottleneck_dim": dann_config.get("bottleneck_dim", None),
        "alpha_max_epochs": dann_config.get("alpha_max_epochs", None),
        "use_binary_loss": dann_config.get("use_binary_loss", False),
        "use_ce_loss": dann_config.get("use_ce_loss", True),
        "truely_mdan": dann_config.get("truely_mdan", True),
        "use_spectral_norm": dann_config.get("use_spectral_norm", True),
        "temperature": dann_config.get("temperature", 1.0),
        "dom_loss_type": dann_config.get("dom_loss_type", "mean"),
        "noise_augmentation": dann_config.get("noise_augmentation", True),
        "normalize_features": dann_config.get("normalize_features", True),
        "emb_regularization": dann_config.get("emb_regularization", None),
        "emb_regularization_weight": dann_config.get("emb_regularization_weight", 0.1),
        "use_afn": dann_config.get("use_afn", False),
        "afn_weight": dann_config.get("afn_weight", 0.01),
        "afn_mode": dann_config.get("afn_mode", "safn"),
        "afn_R": dann_config.get("afn_R", 25.0),
        "afn_delta_r": dann_config.get("afn_delta_r", 1.0),
        "mean_pool_for_discriminator": dann_config.get("mean_pool_for_discriminator", True),
        "softmax_w_detach": dann_config.get("softmax_w_detach", False),
        "non_blank_weighting": dann_config.get("non_blank_weighting", False),
        "binary_cross_entropy_loss": dann_config.get("binary_cross_entropy_loss", False),
        "phase": dann_config.get("phase", 16),
        "weighted_ctc_loss": dann_config.get("weighted_ctc_loss", False),
        "kl_phone_prior_loss_weight": dann_config.get("kl_phone_prior_loss_weight", 0.0),
        "reduce_entropy_loss_weight": dann_config.get("reduce_entropy_loss_weight", 0.0),
        "domain_entropy_weight": dann_config.get("domain_entropy_weight", 0.0),
        "normalize_features_for_discriminator": dann_config.get("normalize_features_for_discriminator", False),
        "include_original": data_config.get("include_original", True),
        "include_stretched_samples": data_config.get("include_stretched_samples", False),
        "include_prolonged_samples": data_config.get("include_prolonged_samples", False),
        "stretch_range": data_config.get("stretch_range", 2.0),
    }

    # Set the CUDA device for this process (torchrun sets LOCAL_RANK)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}, Local Rank {local_rank}] Using device: {args['device']}")
    
    if rank == 0:
        print(f"Using dataset: {args['datasetPath']}")
        print(f"📄 Configuration loaded from: {config_path}")
        if RESUME_FROM_EPOCH > 0:
            print(f"🔄 RESUMING TRAINING from epoch {RESUME_FROM_EPOCH}")
            print(f"📁 Loading checkpoint from: {CHECKPOINT_DIR}")
        if os.path.exists(args['outputDir']):
            print(f"Output directory '{args['outputDir']}' already exists. Press 'c' to continue.")
    
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    
    # Instantiate model
    model = BiT_Phoneme(
        patch_size=args['patch_size'],
        dim=args['dim'],
        dim_head=args['dim_head'],
        nClasses=args['nClasses'],
        nClasses_2=args['nClasses_2'],
        depth=args['depth'],
        heads=args['heads'],
        mlp_dim_ratio=args['mlp_dim_ratio'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        gaussianSmoothWidth=args['gaussianSmoothWidth'],
        T5_style_pos=args['T5_style_pos'],
        max_mask_pct=args['max_mask_pct'],
        num_masks=args['num_masks'], 
        mask_token_zeros=args['mask_token_zero'], 
        num_masks_channels=args['num_masks_channels'], 
        max_mask_channels=args['max_mask_channels'], 
        dist_dict_path=args['dist_dict_path'], 
        consistency=args['consistency'], 
        bottleneck_dim=args['bottleneck_dim'],
        normalize_features=args['normalize_features'],
    )
    
    # Move model to correct device
    model = model.to(args['device'])
    
    # If distributed, wrap model with DDP
    if dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    
    # Train
    trainModelDANN(args, model)



