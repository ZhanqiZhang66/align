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

from neural_decoder.neural_decoder_trainer import trainModel

from neural_decoder.bit import BiT_Phoneme

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
    'obi_log_held_out_interleaved_no_april': os.path.join(BASE_PATHS['obi'], 'ptDecoder_ctc_both_held_out_days_interleaved_no_april'),
    'leia': os.path.join(BASE_PATHS['leia'], 'data'),
    'leia_log': os.path.join(BASE_PATHS['leia'], 'data_log_both'),
    'leia_log_char': os.path.join(BASE_PATHS['leia'], 'data_log_both_char'),
    'leia_log_held_out': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days'), 
    'leia_log_held_out_1': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_1'), 
    'leia_log_held_out_2': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_2'),
    'leia_log_char_phoneme': os.path.join(BASE_PATHS['leia'], 'data_log_both_char_and_phoneme'),
    'leia_log_held_out_interleaved_no_april': os.path.join(BASE_PATHS['leia'], 'data_log_both_held_out_days_interleaved_no_april'),
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Transformer model')
parser.add_argument('--config', type=str, 
                    default='scripts/nrp/train_transformer_cross_session_config.yaml',
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
    
    # Convert lists to tuples where needed
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
        'nClasses_2': model_config.get('nClasses_2'),  # set to None if only one output head 
        'whiteNoiseSD': data_config.get('whiteNoiseSD', 0.8),
        'gaussianSmoothWidth': data_config.get('gaussianSmoothWidth', 2.0),
        'constantOffsetSD': data_config.get('constantOffsetSD', 0.2),
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
        'num_masks_channels': masking_config.get('num_masks_channels', 4),  # number of masks per grid
        'max_mask_channels': masking_config.get('max_mask_channels', 4),  # maximum number of channels to mask per mask
        'max_mask_pct': masking_config.get('max_mask_pct', 0.075), 
        'num_masks': masking_config.get('num_masks', 20),
        'dist_dict_path': masking_config.get('dist_dict_path', '/home3/skaasyap/willett/outputs/dist_dict.pt'), 
        'consistency': masking_config.get('consistency', False), 
        'consistency_scalar': masking_config.get('consistency_scalar', 0.2),
        'wandb_project': other_config.get('wandb_project', 'brain2text-debug'),
        'wandb_entity': other_config.get('wandb_entity', 'victoriazhang-projects'),
        'wandb_name': model_name_base  # WandB run name
    }

    # Set the CUDA device for this process (torchrun sets LOCAL_RANK)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}, Local Rank {local_rank}] Using device: {args['device']}")
        print(f"Using dataset: {args['datasetPath']}")
        print(args)
    
    if rank == 0:
        print(f"Using dataset: {args['datasetPath']}")
        print(f"📄 Configuration loaded from: {config_path}")
        if RESUME_FROM_EPOCH > 0:
            print(f"🔄 RESUMING TRAINING from epoch {RESUME_FROM_EPOCH}")
            print(f"📁 Loading checkpoint from: {CHECKPOINT_DIR}")
        # Warn if output directory exists
        if os.path.exists(args['outputDir']):
            print(f"Output directory '{args['outputDir']}' already exists. Press 'c' to continue.")
            breakpoint()
        
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
        consistency = args['consistency']
    ).to(args['device'])

    # Load pretrained model if specified (before DDP wrapping)
    if args['load_pretrained_model']:
        ckpt_path = os.path.join(args['load_pretrained_model'], 'modelWeights')
        model.load_state_dict(torch.load(ckpt_path, map_location=args['device']), strict=True)
        if rank == 0:
            print(f"Loaded pretrained model from {ckpt_path}")
    
    # Wrap model in DDP if using distributed training
    is_distributed = dist.is_initialized()
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if rank == 0:
            print(f"✅ Model wrapped in DistributedDataParallel")
    
    # Train - pass model.module if DDP, else model
    trainModel(args, model)
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
