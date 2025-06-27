import logging
import os
import random
import shutil
import sys
import pickle
import copy
import torch
from tqdm import tqdm
import copy
import torch.multiprocessing as mp
import datasets
import numpy as np
import evaluate
import transformers
from datasets import Dataset, load_dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from functools import partial

from transformers import (
    HfArgumentParser,
    set_seed,
)
# from transformer4planning.models.model import build_models
from transformer4planning.models.backbone.str_base import build_models
from transformer4planning.utils.args import (
    ModelArguments, 
    DataTrainingArguments, 
    ConfigArguments, 
    PlanningTrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import (PlanningTrainer, CustomCallback)
from torch.utils.data import DataLoader
from transformers.trainer_callback import DefaultFlowCallback
from transformer4planning.trainer import compute_metrics
from datasets import Dataset, load_from_disk

# Pseudo dataset for testing
def make_pseudo_dataset(dataset_size=100):
    # torch tensor で作成
    single_data = {
        "high_res_raster": torch.randint(0, 2, (224, 224, 58)),
        "low_res_raster": torch.randint(0, 2, (224, 224, 58)),
        "context_actions": torch.randint(0, 2, (4, 4)),
        "trajectory_label": torch.randint(0, 2, (80, 4)),
    }

    dataset = Dataset.from_list([single_data for _ in range(dataset_size)])
    return dataset
    
class AutowareCollator:
    def __init__(self, device):
        self.device = device

    def __call__(self, features):
       return {
            "high_res_raster": torch.stack([torch.Tensor(f["high_res_raster"]) / 255.0 for f in features]).to(self.device),
            "low_res_raster": torch.stack([torch.Tensor(f["low_res_raster"]) / 255.0 for f in features]).to(self.device),
            "context_actions": torch.stack([torch.Tensor(f["context_actions"]) for f in features]).to(self.device),
            "trajectory_label": torch.stack([self._preprocess_step(torch.Tensor(f["trajectory_label"])) for f in features]).to(self.device),
            "ego_info": torch.stack([torch.Tensor(f["ego_info"]) for f in features]).to(self.device),
        }
    
    @staticmethod
    def _preprocess_step(step):
        """
        step: (x, y, z, yaw)
        """
        step = torch.cat([step[..., :2] / 100.0, step[..., 2:]], dim=-1)
        return step



def main():
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoware_collator = AutowareCollator(device)
    # dataset = make_pseudo_dataset()
    dataset = load_from_disk("/groups/gcd50654/tier4/dataset/rosbag_full_dataset_new")
    print(dataset)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(train_dataset)
    print(eval_dataset)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, config_args, training_args = parser.parse_args_into_dataclasses()
    # set default label names
    training_args.label_names = ['trajectory_label']
    # training_args.max_steps = 100
    if training_args.gradient_checkpointing:
        """
        Gradient checkpointing is going to crush your training for unknown reasons of the transformers library.
        This problem bugs universally over all backbones and types of encoders!
        See https://discuss.huggingface.co/t/enabling-gradient-checkpointing-and-deepspeed-zero3-raise-train-failure/53789
        """
        logger.warning("Gradient checkpointing is likely going to crush your training for unknown reasons!!!!!")

    # pre-compute raster channels number
    if model_args.raster_channels == 0:
        road_types = 20
        agent_types = 8
        traffic_types = 4
        past_sample_number = int(2 * 20 / model_args.past_sample_interval)  # past_seconds-2, frame_rate-20
        if 'auto' not in model_args.model_name:
            # will cast into each frame
            if model_args.with_traffic_light:
                model_args.raster_channels = 1 + road_types + traffic_types + agent_types
            else:
                model_args.raster_channels = 1 + road_types + agent_types

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # set seed
    set_seed(training_args.seed)
    from datasets import disable_caching
    disable_caching()

    # build model
    model = build_models(model_args)

    # use sync normal
    if model_args.sync_norm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=autoware_collator
    )

    trainer.train()
    
    # save model
    model.save_pretrained(training_args.output_dir)


    # Prediction
    model.eval()
    test_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=autoware_collator, pin_memory=False)
    predictions = model.generate(**next(iter(test_dataloader)))
    print(predictions)

if __name__ == "__main__":
    # 'spawn'を使用するように設定
    mp.set_start_method('spawn')
    main()