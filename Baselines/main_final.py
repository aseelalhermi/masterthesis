import argparse
import json
import wandb
from data_loader import load_dataset, load_dataset_tokenized
from train_test import tune_model, train_and_evaluate_models, tune_model_cv
from train_test_tokenized import tune_model_tokenized, train_and_evaluate_models_tokenized, tune_model_tokenized_cv
from optuna.samplers import TPESampler
import os
import joblib  # or use pickle
import numpy as np
import optuna

def main():
    parser = argparse.ArgumentParser(description="Survival Analysis Pipeline")
    parser.add_argument("--dataset", type=str, choices=["metabric", "support", "gbsg", "flchain"], required=True)
    parser.add_argument("--models", type=str, nargs='+', choices=["cox", "rsf", "gb", "transformer", "deephit", "deepsurv", "deephit_transformer"], required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_runs", type=int, default=10)

    args = parser.parse_args()

    wandb.init(project="survival_analysis", name=f"{args.dataset}_runs")

    # Separate classical and deep models
    classical_models = {"cox", "rsf", "gb"}
    deep_models = {"transformer", "deephit", "deepsurv", "deephit_transformer"}

    selected_models = set(args.models)
    classical_selected = list(selected_models & classical_models)
    deep_selected = list(selected_models & deep_models)
    best_params = {}

    # === Classical models (Cox, RSF, GB)
    if classical_selected:
        x_train, x_val, x_test, survival_train, durations_val, events_val, survival_test, x_train_val, durations_train_val, events_train_val = load_dataset(args.dataset)

        def make_objective_classical(model_type):
            def objective(trial):

                NUM_DURATIONS = 3
                all_times = np.concatenate([
                      survival_train['time'],
                      durations_val,
                      survival_test['time']
                  ])

                all_events =  np.concatenate([
                                survival_train['event'],
                                events_val,
                                survival_test['event']
                            ])           

                quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
                time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)
                survival_val = {"time": durations_val, "event": events_val}
                return tune_model(trial, model_type, x_train, survival_train, x_val, survival_val, time_grid)
            return objective
        
        def make_objective_cv(model_type, x_trainval, durations_trainval, events_trainval):
          def objective(trial):

              NUM_DURATIONS = 3
              #print(y_train)
              all_times = np.concatenate([
                    survival_train['time'],
                    durations_val,
                    survival_test['time']
                ])

              all_events =  np.concatenate([
                              survival_train['event'],
                              events_val,
                              survival_test['event']
                          ])           

              quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
              time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)

              return tune_model_cv(
                  trial=trial,
                  model_type=model_type,
                  x=x_trainval,
                  durations=durations_trainval,
                  events=events_trainval,
                  time_grid=time_grid,
                  n_splits=5
              )
          return objective
        
         for model_type in classical_selected:
            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=10))
            def safe_objective(trial):
                try:
                    return make_objective_classical(model_type)(trial)
                except (ValueError, RuntimeError) as e:
                    print(f"[Trial Pruned] {model_type} - Error: {e}")
                    raise optuna.exceptions.TrialPruned()
            study.optimize(safe_objective, n_trials=args.n_trials)
            best_params[model_type] = study.best_params
    
    # === Deep models (Transformer, DeepSurv, DeepHit)
    if deep_selected:
        x_train_num, x_val_num, x_test_num, x_train_cat, x_val_cat, x_test_cat, survival_train, durations_val, events_val, survival_test, cat_cardinalities, x_train_val_num, x_train_val_cat, durations_train_val, events_train_val = load_dataset_tokenized(args.dataset, args.custom_data_path)
        
        def make_objective_deep(model_type):
            def objective(trial):

                NUM_DURATIONS = 3

                all_times = np.concatenate([
                      survival_train['time'],
                      durations_val,
                      survival_test['time']
                  ])

                all_events =  np.concatenate([
                                survival_train['event'],
                                events_val,
                                survival_test['event']
                            ])           

                quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
                time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)
                survival_val = {"time": durations_val, "event": events_val}
                return tune_model_tokenized(trial, model_type, x_train_num, x_train_cat, survival_train, x_val_num, x_val_cat, survival_val, time_grid, cat_cardinalities)
            return objective
        
        def make_objective_deep_cv(model_type,  x_train_val_num, x_train_val_cat, durations_train_val, events_train_val):
            def objective(trial):
                NUM_DURATIONS = 3
                #print(y_train)
                all_times = np.concatenate([
                      survival_train['time'],
                      durations_val,
                      survival_test['time']
                  ])

                all_events =  np.concatenate([
                                survival_train['event'],
                                events_val,
                                survival_test['event']
                            ])           


                quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
                time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)

                return tune_model_tokenized_cv(
                  trial=trial,
                  model_type=model_type,
                  x_train_val_num=x_train_val_num, 
                  x_train_val_cat=x_train_val_cat, 
                  durations=durations_train_val, 
                  events=events_train_val,
                  time_grid=time_grid,
                  cat_cardinalities=cat_cardinalities,
                  n_splits=5
              )

            return objective


        for model_type in deep_selected:
            
            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=10))
            def safe_objective(trial):
                try:
                    return make_objective_deep(model_type)(trial)
                except (ValueError, RuntimeError) as e:
                    print(f"[Trial Pruned] {model_type} - Error: {e}")
                    raise optuna.exceptions.TrialPruned()
            study.optimize(safe_objective, n_trials=args.n_trials)
            best_params[model_type] = study.best_params

    # === Evaluation

    NUM_DURATIONS = 3

    all_times = np.concatenate([
          survival_train['time'],
          durations_val,
          survival_test['time']
      ])

    all_events =  np.concatenate([
                    survival_train['event'],
                    events_val,
                    survival_test['event']
                ])           

    quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
    time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)

    all_eval_results = {}

    if classical_selected:
        trained_models, eval_results = train_and_evaluate_models(
            classical_selected, best_params, x_train, x_test, survival_train, survival_test, args.n_runs, time_grid
        )
        all_eval_results.update(eval_results)
        os.makedirs("saved_models", exist_ok=True)
        for model_name, model in trained_models.items():
            joblib.dump(model, f"saved_models/{args.dataset}_{model_name}.pkl")

    if deep_selected:
        eval_results = train_and_evaluate_models_tokenized(
            deep_selected, best_params, x_train_num, x_train_cat, x_test_num, x_test_cat, survival_train, survival_test, cat_cardinalities, time_grid, args.n_runs
        )
        all_eval_results.update(eval_results)
        os.makedirs("saved_models", exist_ok=True)

    wandb.log({"Final Evaluation": all_eval_results})
    wandb.finish()
    print("Final Results:")
    print(json.dumps(all_eval_results, indent=4))

if __name__ == "__main__":
    main()

