#!/bin/bash
#SBATCH --job-name=LeRobot
#SBATCH --output=logs/train-pi0.%j.out
#SBATCH --error=logs/train-pi0.%j.err

#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -t 2:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --hint=nomultithread

echo ${SLURM_NODELIST}

module purge
module load arch/h100
module load miniforge/25.9.1
conda activate lerobot

export WANDB_MODE=offline
export WANDB_API_KEY=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
export HF_HOME=$SCRATCH/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_LEROBOT_HOME=$SCRATCH/.cache/huggingface/lerobot
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME=$SCRATCH/.cache/torch

lerobot-train \
        --job_name=pi0_base_pick_and_place-200k_steps-bs_8-lr_5e-5 \
        --dataset.repo_id="clementPhd/FrankaWujiPickandPlace" \
        --dataset.root="/lustre/fsn1/projects/rech/iyg/uvx44rt/.cache/huggingface/lerobot/hub/datasets--clementPhd--FrankaWujiPickandPlace/snapshots/d74d0c68ee5de203ded469d2f8180da6b98e71e6" \
        --policy.path="/lustre/fswork/projects/rech/iyg/uvx44rt/mylerobot/pi0_ckpt/pi0_base_local/" \
        --policy.push_to_hub=false \
        --policy.device=cuda \
        --policy.chunk_size=20 \
        --policy.n_action_steps=20 \
        --rename_map='{"observation.images.world_camera": "observation.images.base_0_rgb", "observation.images.wrist_camera": "observation.images.left_wrist_0_rgb"}' \
        --policy.scheduler_decay_steps=200_000 \
        --policy.optimizer_lr=5e-5 \
        --policy.dtype=bfloat16 \
        --batch_size=8 \
        --steps=200_000 \
        --save_freq=200_000 \
        --wandb.enable=true \
        --wandb.entity=brunomachadocarneiro37 \
        --wandb.disable_artifact=true \
        --wandb.mode=offline \
        --wandb.add_tags=false