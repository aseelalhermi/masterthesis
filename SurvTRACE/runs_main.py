import optuna
import numpy as np
import torch
import wandb
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from survtrace.train_utils import Trainer
from survtrace.evaluate_utils import Evaluator
from survtrace.dataset import load_data
from survtrace.model import SurvTraceSingle
from survtrace.config import STConfig
from survtrace.utils import set_random_seed
from survtrace.evaluate_utils import plot_survival_analysis
import pandas as pd
from copy import deepcopy
def objective(trial, base_config):
  try:
    config = deepcopy(base_config)
    config.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    config.intermediate_size = trial.suggest_categorical("intermediate_size", [32, 64, 128])
    config.num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)
    config.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    config.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    config.num_attention_heads=trial.suggest_categorical("num_attention_heads", [2, 4, 8])
    config.attention_probs_dropout_prob=trial.suggest_float("attention_probs_dropout_prob", 0.1, 0.5)
    config.hidden_dropout_prob=trial.suggest_float("hidden_dropout_prob", 0.1, 0.5)

    set_random_seed(config.seed)

    # Load full train/val/test split and transformed labels
    df, X_train, y_train_trans, X_test, y_test_raw, X_val, y_val_trans, y_val_raw, X_temp, y_trainval_trans, y_trainval_raw, y_temp_trans, df_trainval = load_data(config)

        
    train_set = (X_train, y_train_trans)
    val_set   = (X_val, y_val_trans)
    eval_val_set = (X_val, y_val_raw)  # for evaluation

    model = SurvTraceSingle(config)
    trainer = Trainer(model)
    train_loss, val_loss = trainer.fit(train_set, val_set,
                batch_size=64, epochs=100,
                learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    evaluator = Evaluator(df, train_index=X_train.index)
    metrics = evaluator.eval(model, eval_val_set)
    ibs = metrics["integrated_brier"]
    if np.isnan(ibs):
        raise optuna.exceptions.TrialPruned()
    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("val_loss", val_loss)
    #return -metrics["c_index_censored"]
    return ibs
  except Exception as e:
    print(f"Trial failed with error: {e}")
    raise optuna.exceptions.TrialPruned()

def run_optuna_then_evaluate_n_seeds(n_trials=100, n_seeds=10, base_config=None):
    sampler = TPESampler(seed=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, base_config), n_trials=n_trials)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    
    all_metrics = []
    for seed in range(n_seeds):
        config = deepcopy(base_config)
        config.update(best_params)
        config.seed = seed

        wandb.init(project="survtrace-optuna", config=config, name=f"seed_{seed}", reinit=True)

        set_random_seed(seed)

        (
            df_all, df_train, y_train_trans, df_test, y_test_raw, df_val, y_val_trans, df_y_val_raw,
            X_temp, y_trainval_trans, y_trainval_raw, y_temp_trans, df_trainval
        ) = load_data(config)

        trainval_set = (X_temp, y_trainval_trans)
        train_set = (df_train, y_train_trans)
        val_set   = (df_val, y_val_trans)
        test_set  = (df_test, y_test_raw)  # <-- raw for test

        model = SurvTraceSingle(config)
        trainer = Trainer(model)
        train_loss, val_loss = trainer.fit(train_set, val_set,
                    batch_size=64, epochs=100,
                    learning_rate=config.learning_rate, weight_decay=config.weight_decay)

        for epoch, (t, v) in enumerate(zip(train_loss, val_loss)):
          wandb.log({"train_loss": t, "val_loss": v, "epoch": epoch})
          
          # Re-run a forward pass with a small batch for visualization
          model.eval()
          with torch.no_grad():
              sample_x = df_train.iloc[:8]
              x_cat = torch.tensor(sample_x.iloc[:, :config.num_categorical_feature].values).long()
              x_num = torch.tensor(sample_x.iloc[:, config.num_categorical_feature:].values).float()


              if model.use_gpu:
                  x_cat = x_cat.cuda()
                  x_num = x_num.cuda()


              outputs = model(input_ids=x_cat, input_nums=x_num)

        evaluator = Evaluator(df_all, train_index=df_train.index)
        metrics = evaluator.eval(model, test_set)

        # Only for plotting
        df_test = df_test.reset_index(drop=True)
        x_test = test_set[0]
        surv = model.predict_surv(x_test, batch_size=64)
        risk = 1 - surv
        durations_test = y_test_raw['duration'].values
        events_test = y_test_raw['event'].values

        plot_survival_analysis(
            surv=surv,
            risk=risk,
            durations_test=durations_test,
            events_test=events_test,
            times=config.duration_index[1:-1],
            horizons=config.horizons,
            metrics=metrics,
        )

        metrics = evaluator.eval(model, test_set)
        wandb.log(metrics)
        wandb.finish()

        all_metrics.append(metrics)

    # Aggregate results
    from collections import defaultdict
    agg_metrics = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            agg_metrics[k].append(v)

    print("\n=== Final Results Across 10 Seeds ===")
    for k, v in agg_metrics.items():
        print(f"{k} - mean: {np.mean(v):.4f}, std: {np.std(v):.4f}")
