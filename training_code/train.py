import os
import numpy as np
import pdb
import cv2

import trainers
import datasets

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from tqdm import tqdm

# Handle setting up arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--resume', action="store_true")
parser.add_argument('--eval', action="store_true")

parser.add_argument('--max_skip', type=int, default=10)
parser.add_argument('--min_skip', type=int, default=1)

parser.add_argument('--train_seg', action="store_true")
parser.add_argument('--train_contour', action="store_true")

parser.add_argument('--train_flow', action="store_true")
parser.add_argument('--train_pos', action="store_true")
parser.add_argument('--single_scale_flow', action="store_true")

parser.add_argument('--input_type', type=str, default="event_volume")

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--experiment_name', type=str, default='test')


# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)
parser = trainers.Net.add_model_specific_args(parser)
parser = datasets.DataModule.add_data_args(parser)

tmp_args, _ = parser.parse_known_args()
orig_args = parser.parse_args()
args = vars(orig_args)
model = trainers.Net(**args)

# build dataset
data_module = datasets.DataModule(**args)

# init logger
logger = TensorBoardLogger("logs", name=tmp_args.experiment_name)
orig_args.logger=logger
trainer = Trainer.from_argparse_args(orig_args)

# start training
trainer.fit(model, data_module)
