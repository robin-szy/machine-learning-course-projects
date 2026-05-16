#!/bin/bash -l
#SBATCH --job-name=ML_traffic_forecast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
# SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=0-01:00:00 #DD-HH:MM:SS
#SBATCH --array=1-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module --force purge
# module load env/development/2025a
# module load lang/Python/3.13.1-GCCcore-14.2.0


source $HOME/deep_learning/DL_env_latest/bin/activate

PROJECT_DIR="$HOME/Traffic_Forecast"
CONFIG_FILE="$PROJECT_DIR/configs.csv"
DATA_DIR="$PROJECT_DIR/data"
RUNS_DIR="$PROJECT_DIR/runs"

cd "$PROJECT_DIR"
mkdir -p "$RUNS_DIR" logs

CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

IFS=',' read -r RUN_NAME SCRIPT SEED <<< "$CONFIG_LINE"

MODEL_FILE="$RUNS_DIR/${RUN_NAME}.pth"

srun python -u "$SCRIPT" \
    --data-dir "$DATA_DIR" \
    --model-file "$MODEL_FILE" \
    --seed "$SEED" \
    --final-train

echo "Finished run: ${RUN_NAME}"
echo "Saved model: ${MODEL_FILE}"




