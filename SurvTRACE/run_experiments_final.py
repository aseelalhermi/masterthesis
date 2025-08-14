import argparse
import wandb
import torch
import numpy as np
from copy import deepcopy
from optuna.samplers import TPESampler

from survtrace.config import STConfig
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.dataset import load_data
from survtrace.train_utils import Trainer
from survtrace.evaluate_utils import Evaluator
from runs_main import run_optuna_then_evaluate_n_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Run SurvTrace Experiment")
    parser.add_argument("--dataset", type=str, default="metabric", choices=["metabric", "support", "flchain", "gbsg"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--add_mask", action='store_true')
    parser.add_argument("--attn", type=str, default="saint", choices=["saint", "regular"])
    parser.add_argument("--pos_enc", type=str, default="none", choices=["absolute", "learnable", "sinosoidal", "rope", "none"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare config
    config = deepcopy(STConfig)
    config.data = args.dataset
    config.add_mask = args.add_mask
    config.use_saint_attention = args.attn == "saint"
    if args.pos_enc == "none":
      config.position_embedding_type = None
    else:
      config.position_embedding_type = args.pos_enc
    run_name = f"{args.dataset}"
    wandb.init(project="survtrace-argparse", config=config, name=run_name)

    run_optuna_then_evaluate_n_seeds(n_trials=args.n_trials, n_seeds=args.n_seeds, base_config=config)

    wandb.finish()


if __name__ == "__main__":
    main()

