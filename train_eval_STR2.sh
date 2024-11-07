#!/bin/bash

export PATH_TO_DATASET_FOLDER=/home/kai.yamashita/projects/datasets/nuplan-v1.1_STR_2
export PATH_TO_OUTPUT_FOLDER=/home/kai.yamashita/projects/StateTransformer/outputs
export PATH_TO_LOG_FOLDER=/home/kai.yamashita/projects/StateTransformer/logs
export MODEL_NAME=scratch-mixtral-800m-deep
export EXPERIMENT_NAME=STR2_${MODEL_NAME}

wandb login 1910ad05118cde66ceb8d2eae5ad56f8a7d46eac


CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run \
--nproc_per_node=2 --master_port 12345 runner.py \
--model_name $MODEL_NAME \
--model_pretrain_name_or_path None \
--saved_dataset_folder ${PATH_TO_DATASET_FOLDER} \
--output_dir ${PATH_TO_OUTPUT_FOLDER} \
--logging_dir ${PATH_TO_LOG_FOLDER} \
--run_name ${EXPERIMENT_NAME} \
--num_train_epochs 20 \
--per_device_train_batch_size 16 \
--warmup_steps 50 \
--weight_decay 0.01 \
--logging_steps 100 \
--save_strategy steps \
--save_steps 6000 \
--dataloader_num_workers 24 \
--dataloader_drop_last True \
--save_total_limit 5 \
--do_train \
--task nuplan \
--remove_unused_columns False \
--do_eval \
--evaluation_strategy epoch \
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
--use_speed \
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
--output_router_logits True \
--augment_method track \
--augment_max_dy 0.5 \
--augment_max_dyaw 0.05 \
--overwrite_output_dir