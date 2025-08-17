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
from survtrace.utils import set_random_seed
from survtrace.evaluate_utils import plot_survival_analysis
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

      (
          df,
          df_train, y_train_trans, y_train_raw,
          df_test, y_test_raw,
          df_val, y_val_trans, y_val_raw,
          attention_mask, X_temp, y_trainval_trans, 
          y_trainval_raw, y_temp_trans, df_trainval
      ) = load_data(config)

      train_set = (df_train, y_train_trans, attention_mask.loc[df_train.index])
      val_set   = (df_val, y_val_trans, attention_mask.loc[df_val.index])
      eval_val_set = (df_val, y_val_raw, attention_mask.loc[df_val.index])  # for evaluation

      model = SurvTraceSingle(config)
      trainer = Trainer(model)
      train_loss, val_loss = trainer.fit(train_set, val_set,
                  batch_size=64, epochs=100,
                  learning_rate=config.learning_rate, weight_decay=config.weight_decay)

      evaluator = Evaluator(df, train_index=df_train.index)
      metrics = evaluator.eval(model, eval_val_set)  # <-- use raw durations/events

      ibs = metrics["integrated_brier"]
      if np.isnan(ibs):
          raise optuna.exceptions.TrialPruned()

      trial.set_user_attr("train_loss", train_loss)
      trial.set_user_attr("val_loss", val_loss)
      return metrics["integrated_brier"]  # Optuna minimizes
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
            df,
            df_train, y_train_trans, y_train_raw,
            df_test, y_test_raw,
            df_val, y_val_trans, y_val_raw,
            attention_mask, X_temp, y_trainval_trans, 
            y_trainval_raw, y_temp_trans, df_trainval
        ) = load_data(config)

        trainval_set = (X_temp, y_trainval_trans, attention_mask.loc[X_temp.index])
        train_set = (df_train, y_train_trans, attention_mask.loc[df_train.index])
        val_set   = (df_val, y_val_trans, attention_mask.loc[df_val.index])
        test_set  = (df_test, y_test_raw, attention_mask.loc[df_test.index])  # <-- raw for test

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
              mask_x = attention_mask.loc[sample_x.index]
              x_cat = torch.tensor(sample_x.iloc[:, :config.num_categorical_feature].values).long()
              x_num = torch.tensor(sample_x.iloc[:, config.num_categorical_feature:].values).float()
              mask = torch.tensor(mask_x.values).float()

              if model.use_gpu:
                  x_cat = x_cat.cuda()
                  x_num = x_num.cuda()
                  mask = mask.cuda()

              outputs = model(input_ids=x_cat, input_nums=x_num, attention_mask=mask, output_attentions=True, output_col_attentions=config.use_saint_attention)
              if len(outputs) >= 4:
                  _, _, row_probs, col_probs = outputs

                  def log_attention_map(attn_map, name):
                      if isinstance(attn_map, tuple):
                        attn_map = attn_map[-1]  # Or average across layers: torch.stack(attn_map).mean(0)
                      if attn_map.ndim == 3:
                          attn_map = attn_map.mean(0)  # Mean over heads → shape (N, N)
                      elif attn_map.ndim == 4:
                          attn_map = attn_map.mean(dim=(0, 1))  # Mean over layers and heads
                      fig, ax = plt.subplots()
                      im = ax.imshow(attn_map.cpu().numpy())  # mean over batch
                      plt.colorbar(im)
                      wandb.log({name: wandb.Image(fig)}, step=epoch)
                      plt.close(fig)
                  
                  log_attention_map(row_probs, "row_attention")
                  if col_probs is not None:
                      log_attention_map(col_probs, "col_attention")


        evaluator = Evaluator(df, train_index=df_train.index)
        metrics = evaluator.eval(model, test_set)

        # Only for plotting
        df_test = df_test.reset_index(drop=True)
        x_test = test_set[0]
        mask_test = attention_mask.loc[x_test.index]
        surv = model.predict_surv(x_test, batch_size=64, attention_mask=mask_test)
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
