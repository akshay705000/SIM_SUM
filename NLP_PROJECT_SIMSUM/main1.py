# -- fix path --
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

import time
import json

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from preprocessor import WIKI_DOC, EXP_DIR
from Bart_baseline_finetuned import BartBaseLineFineTuned, train
#from T5_baseline_finetuned import T5BaseLineFineTuned, train

def parse_arguments():
    p = ArgumentParser()
    p.add_argument('--seed', type=int, default=42, help='randomization seed')

    # Add model specific arguments
    #p = T5BaseLineFineTuned.add_model_specific_args(p)
    p = BartBaseLineFineTuned.add_model_specific_args(p)

    # Parse arguments
    args, _ = p.parse_known_args()
    return args

# Create experiment directory
def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = {key: str(value) for key, value in kwargs.items()}
    json.dump(kwargs_str, filepath.open('w'), indent=4)

def run_training(args, dataset):
    args.output_dir = get_experiment_dir(create_dir=True)
    # Logging the args
    log_params(args.output_dir / "params.json", vars(args))

    args.dataset = dataset
    print("Dataset:", args.dataset)
    train(args)

dataset = WIKI_DOC
args = parse_arguments()
run_training(args, dataset)

