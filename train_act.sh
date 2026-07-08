#!/bin/sh  
#SBATCH --job-name=LeRobot                                                                                                              #SBATCH --job-name=LeRobot
#SBATCH --output=logs/train-act.%j.out
#SBATCH --error=logs/train-act.%j.err

#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -t 2:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread
#SBATCH --mail-user=clement.prigent@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

module purge
module load arch/h100
module load miniforge/25.9.1
conda activate lerobot

export WANDB_MODE=offline
export WANDB_API_KEY=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

lerobot-train \
  --dataset.repo_id="/lustre/fsn1/projects/rech/iyg/uvx44rt/lerobot_datasets" \
  --job_name=act_base-pick-and-place \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=False \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.optimizer_lr=8e-5 \
  --policy.optimizer_weight_decay=1e-2 \
  --policy.optimizer_lr_backbone=8e-5 \
  --batch_size=448 \
  --num_workers=8 \
  --steps=300_000 \
  --save_freq=100_000 \
  --wandb.enable=true \
  --wandb.entity=Clémentppr \
  --wandb.disable_artifact=true \
  --wandb.mode=offline \
  --wandb.add_tags=false