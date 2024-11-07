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
import multiprocessing as mp
import datasets
import numpy as np
import evaluate
import transformers
from datasets import Dataset
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

from datasets import Dataset, Value

from runner import load_dataset

import pickle

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, config_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if model_args.task == "nuplan":
        all_maps_dic = {}
        map_folder = os.path.join(data_args.saved_dataset_folder, 'map')
        for each_map in os.listdir(map_folder):
            if each_map.endswith('.pkl'):
                map_path = os.path.join(map_folder, each_map)
                with open(map_path, 'rb') as f:
                    map_dic = pickle.load(f)
                map_name = each_map.split('.')[0]
                all_maps_dic[map_name] = map_dic

    index_root = os.path.join(data_args.saved_dataset_folder, 'index')

    test_dataset = load_dataset(index_root, "test", data_args.dataset_scale, data_args.agent_type, False)

    test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=1, 
                pin_memory=True,
                drop_last=True
            )
    
    sample_batch = next(iter(test_dataloader))
    
    with open("before_collate.pkl", 'wb') as f:
        pickle.dump(sample_batch, f)
    
    print("Now before_collate Log")
    for key, value in sample_batch.items():
        print("*" * 300)
        print("Key:", key)
        if hasattr(value, "shape"):
            print("Value Shape:", value.shape)
        else:
            print("Value", value)
        print("*" * 300)
    print("-" * 500)

    from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
    collate_fn = partial(nuplan_rasterize_collate_func,
                            dic_path=data_args.saved_dataset_folder,
                            all_maps_dic=all_maps_dic,
                            **model_args.__dict__)

    test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )
    
    sample_batch = next(iter(test_dataloader))

    with open("after_collate.pkl", 'wb') as f:
        pickle.dump(sample_batch, f)

    print("Now after_collate Log")
    for key, value in sample_batch.items():
        print("*" * 300)
        print("Key:", key)
        if hasattr(value, "shape"):
            print("Value Shape:", value.shape)
        else:
            print("Value", value)
        print("*" * 300)
    print("-" * 500)


if __name__ == "__main__":
    main()
    



