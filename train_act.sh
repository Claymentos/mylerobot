                                                                                                                 
#!/bin/bash
#SBATCH --job-name=LeRobot
#SBATCH --output=logs/train-act.%j.out
#SBATCH --error=logs/train-act.%j.err

#SBATCH -A yfl@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH -t 100:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --hint=nomultithread

echo ${SLURM_NODELIST}

module purge
module load arch/h100

source /linkhome/rech/genkmw01/uuv83ah/.bashrc
conda activate mylerobot

export WANDB_MODE=offline

lerobot-train \
  --dataset.repo_id="/lustre/fsn1/projects/rech/gjs/uuv83ah/lerobot_datasets/BrunoM42/shelf_picking" \
  --job_name=act_base-shelf_picking \
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
  --policy.use_env_state_feature=false \
  --batch_size=448 \
  --num_workers=8 \
  --steps=300_000 \
  --save_freq=100_000 \
  --wandb.enable=true \
  --wandb.entity=brunomachadocarneiro37 \
  --wandb.disable_artifact=true \
  --wandb.mode=offline \
  --wandb.add_tags=false