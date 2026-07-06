#!/bin/bash
#SBATCH -p kempner_requeue # partition
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --job-name=speechbci     # create a short name for your job
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --cpus-per-task=16       # number of cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G               # total memory per node (16 GB per cpu-core is default)
#SBATCH --time=0-02:00:00        # total run time limit (HH:MM:SS)
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user shunli@g.harvard.edu

# Run the job on cluster
# cd SpeechBCI
# sbatch unzip_data.sh

# Configuration: auto-detect from submit dir (SLURM) or script location; override via project_dir env var
project_dir="${project_dir:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
export project_dir
# script_name="format_competition_data"
script_name="format_competition_data_conditions"

# Set datapath and analysispath
script_path="scripts/${script_name}.py"
output_dir="${project_dir}/outputs/${SLURM_JOB_ID}-$(date +%Y%m%d)-${script_name}"
slurm_output_dir="${project_dir}/slurm_outputs/${SLURM_JOB_ID}-$(date +%Y%m%d)-${script_name}"

# Load modules
# module load python/3.9.23-fasrc01
module load Mambaforge/23.11.0-fasrc01

# Initialize conda/mamba (needed to properly activate environments in SLURM)
eval "$(conda shell.bash hook)"

# Activate environment
# source ~/.bashrc
mamba_env="speechbci"
mamba activate $mamba_env

# Verify we're using the correct Python
echo "=== Python Environment Info ==="
echo "Active mamba env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "==============================="

# Create output directory
mkdir -p "$output_dir"
mkdir -p "$slurm_output_dir"

# Run script
cd "$project_dir"
python $script_path

# python scripts/format_competition_data_conditions.py --data-dir data/T12/competitionData --data-save-base data/T12/ptDecoder_ctc_both
# python scripts/format_competition_data_conditions.py --conditions 0
# python scripts/format_competition_data_conditions_t15.py
python scripts/format_competition_data_conditions.py \
  --data-dir /n/holylfs06/LABS/bsabatini_lab/Lab/shunnnli/speechbci/data/T15/hdf5_data_final \
  --output-dir /n/holylfs06/LABS/bsabatini_lab/Lab/shunnnli/speechbci/data/T15/ptDecoder_ctc_both \
  --conditions 5

# Move job outputs to experiment directory
# Move job outputs to experiment directory
mv $project_dir/$SLURM_JOB_ID.out $slurm_output_dir
mv $project_dir/$SLURM_JOB_ID.err $slurm_output_dir

# Remove compressed data files
# tar -xvf data/T12/languageModel.tar.gz
# echo "Unpacked languageModel.tar.gz"
# rm data/T12/languageModel.tar.gz
# tar -xvf data/T15/languageModel_5gram.tar.gz
# echo "Unpacked languageModel_5gram.tar.gz"
# rm data/T15/languageModel_5gram.tar.gz