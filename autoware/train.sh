#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=3:00:00
#PBS -P gcd50654
#PBS -k oe
#PBS -j oe

export MODEL_NAME=scratch-mixtral-800m

export OUTPUT_ROOT="/groups/gcd50654/tier4/kai-yamashita/latest-${MODEL_NAME}"

export PATH_TO_OUTPUT_FOLDER="$OUTPUT_ROOT/output"
export PATH_TO_LOG_FOLDER="$OUTPUT_ROOT/log"
mkdir -p $PATH_TO_OUTPUT_FOLDER
mkdir -p $PATH_TO_LOG_FOLDER

# raster channel 34なのは，num_route_channel 2 + num_road_type 20 + num_traffic_light 4 + num_agent_type 8 であるから
# 入力データの実際のchannel数が58なのは，num_route_channel 2 + num_road_type 20 + num_traffic_light 4 + num_agent_type 8 * len(context_actions) 4であるから

module load cuda cudnn
export EXPERIMENT_NAME=autoware_trial_${MODEL_NAME}
# 割り当てられたGPUの情報を取得
GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader)

# # CUDA_VISIBLE_DEVICESを設定
# export CUDA_VISIBLE_DEVICES=$(echo $GPU_IDS | tr '\n' ',' | sed 's/,$//')
source /home/acf15382lp/projects/STR2/.venv/bin/activate

export WANDB_PROJECT=autoware_sample
export EXPERIMENT_NAME=autoware_trial_${MODEL_NAME}

cd /home/acf15382lp/projects/STR2/autoware_trial && \
torchrun --nproc_per_node=8 train.py \
--model_name $MODEL_NAME \
--model_pretrain_name_or_path None \
--saved_dataset_folder hogehoge \
--output_dir ${PATH_TO_OUTPUT_FOLDER} \
--logging_dir ${PATH_TO_LOG_FOLDER} \
--run_name ${EXPERIMENT_NAME} \
--num_train_epochs 100 \
--per_device_train_batch_size 4 \
--warmup_steps 1 \
--weight_decay 0.01 \
--logging_steps 5 \
--logging_strategy steps \
--save_strategy steps \
--save_steps 5000 \
--dataloader_num_workers 1 \
--dataloader_drop_last False \
--do_train \
--task nuplan \
--remove_unused_columns False \
--evaluation_strategy no \
--per_device_eval_batch_size 8 \
--predict_yaw True \
--use_proposal 0 \
--selected_exponential_past True \
--mean_circular_loss True \
--raster_channels 34 \
--use_mission_goal False \
--raster_encoder_type vit \
--vit_intermediate_size 768 \
--lr_scheduler_type cosine_with_restarts \
--num_cycles 10 \
--use_key_points specified_backward \
--augment_current_pose_rate 0.5 \
--augment_current_with_past_linear_changes True \
--augment_index 5 \
--attn_implementation flash_attention_2 \
--sync_norm True \
--bf16 True \
--max_sim_samples 64 \
--inspect_kp_loss \
--kp_dropout 0.1 \
--report_to wandb \
--run_name ${EXPERIMENT_NAME} \
--output_router_logits True \
--augment_method linear \
--augment_max_dy 0.5 \
--augment_max_dyaw 0.05 \
--dataloader_pin_memory False \
--overwrite_output_dir