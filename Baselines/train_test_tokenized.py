import numpy as np
import pandas as pd
import wandb
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc, integrated_brier_score, concordance_index_censored
from pycox.models import CoxPH as PyCoxPH, DeepHitSingle
from custom_models import TransformerSurvivalModel2, get_deepsurv_net2, get_deephit_net2, get_optimizer, DeepHitTransformerSurvivalModel2

# Evaluation Helpers

def get_survival_probs_manual(surv_df, times):
    times = np.asarray(times)
    time_grid = surv_df.index.values
    surv_values = surv_df.values
    return pd.DataFrame(
        np.stack([
            [np.interp(t, time_grid, surv_values[:, i]) for t in times]
            for i in range(surv_values.shape[1])
        ]),
        columns=[f"S(t={t})" for t in times]
    )

class SurvivalModelWrapper:
    def __init__(self, model, model_type, time_grid):
        self.model = model
        self.model_type = model_type
        self.times = time_grid

    def predict_surv_df(self, x):
        surv_df = self.model.predict_surv_df(x)
        return get_survival_probs_manual(surv_df, self.times)

def train_transformer(x_train_num, x_train_cat, survival_train, time_grid, cat_cardinalities, params):
    net = TransformerSurvivalModel2(
        hidden_size=params['hidden_size'],
        num_heads=params['num_attention_heads'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout'],
        num_num_features=x_train_num.shape[1],
        cat_cardinalities=cat_cardinalities
    )
    optimizer = get_optimizer(net, params['optimizer'], params['lr'], params['weight_decay'])
    model = PyCoxPH(net, optimizer)
    durations = survival_train["time"].copy()
    events = survival_train["event"].astype(bool).copy()
    model.fit((x_train_num, x_train_cat), (durations, events), epochs=params['epochs'])
    model.compute_baseline_hazards((x_train_num, x_train_cat), (durations, events))
    return model

def train_deephit_transformer(x_train_num, x_train_cat, survival_train, time_grid, cat_cardinalities, params, num_durations):
    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train = labtrans.fit_transform(survival_train["time"], survival_train["event"])
    net = DeepHitTransformerSurvivalModel2(
        hidden_size=params['hidden_size'],
        num_heads=params['num_attention_heads'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout'],
        num_num_features=x_train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        num_durations=num_durations
    )

    model = DeepHitSingle(net, get_optimizer(net, params['optimizer'], params['lr']),
                          alpha=params['alpha'], sigma=params['sigma'], duration_index=labtrans.cuts)
    durations = survival_train["time"].copy()
    events = survival_train["event"].astype(bool).copy()
    model.fit((x_train_num, x_train_cat), y_train, batch_size=64, epochs=params['epochs'], verbose=False)
    return model

def train_deepsurv(x_train_num, x_train_cat, survival_train, cat_cardinalities, params):
    net = get_deepsurv_net2(
        input_dim_num=x_train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_size=params["hidden_size"],
        num_nodes1=params['num_nodes1'],
        num_nodes2=params['num_nodes2'],
        dropout=params['dropout']
    )
    optimizer = get_optimizer(net, params['optimizer'], params['lr'])
    model = PyCoxPH(net, optimizer)
    durations = survival_train["time"].copy()
    events = survival_train["event"].astype(bool).copy()
    model.fit((x_train_num, x_train_cat), (durations, events), epochs=params['epochs'])
    model.compute_baseline_hazards((x_train_num, x_train_cat), (durations, events))
    return model

def train_deephit(x_train_num, x_train_cat, survival_train, cat_cardinalities, params, NUM_DURATIONS):
    labtrans = DeepHitSingle.label_transform(NUM_DURATIONS)
    y_train = labtrans.fit_transform(survival_train["time"], survival_train["event"])
    net = get_deephit_net2(
        input_dim_num=x_train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        hidden_size=params["hidden_size"],
        num_nodes=params["num_nodes"],
        dropout=params["dropout"],
        num_durations=NUM_DURATIONS
    )
    model = DeepHitSingle(net, get_optimizer(net, params['optimizer'], params['lr']),
                          alpha=params['alpha'], sigma=params['sigma'], duration_index=labtrans.cuts)
    model.fit((x_train_num, x_train_cat), y_train, batch_size=64, epochs=params['epochs'], verbose=False)
    return model



def tune_model_tokenized_cv(trial, model_type, x_train_val_num, x_train_val_cat, durations, events, time_grid, cat_cardinalities, n_splits=5):
    from sksurv.util import Surv
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(x_train_val_num, events):
        x_train_num, x_val_num = x_train_val_num[train_idx], x_train_val_num[val_idx]
        x_train_cat, x_val_cat = x_train_val_cat[train_idx], x_train_val_cat[val_idx]
        d_train, d_val = durations[train_idx], durations[val_idx]
        e_train, e_val = events[train_idx], events[val_idx]
        surv_train = Surv.from_arrays(e_train, d_train)
        surv_val = Surv.from_arrays(e_val, d_val)

        score = tune_model_tokenized(
            trial=trial,
            model_type=model_type,
            x_train_num=x_train_num,
            x_train_cat=x_train_cat,
            survival_train={"time": d_train, "event": e_train},
            x_val_num=x_val_num, 
            x_val_cat=x_val_cat, 
            survival_val={"time": d_val, "event": e_val},
            time_grid=time_grid, 
            cat_cardinalities=cat_cardinalities
        )
        scores.append(score)

    return np.mean(scores)

def tune_model_tokenized(trial, model_type, x_train_num, x_train_cat, survival_train, x_val_num, x_val_cat, survival_val, time_grid, cat_cardinalities):
    NUM_DURATIONS = 3
    HORIZONS = [0.25, 0.5, 0.75]
    et_val = Surv.from_arrays(survival_val["event"], survival_val["time"])

    if model_type == "transformer":

        params = {
            'hidden_size': trial.suggest_categorical("hidden_size", [8, 16, 32]),
            'num_attention_heads': trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
            'num_layers': trial.suggest_int("num_layers", 2, 4),
            'dim_feedforward': trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
            'dropout': trial.suggest_float("dropout", 0.0, 0.5),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float("weight_decay", 0.0, 0.3),
            'epochs': trial.suggest_int("epochs", 10, 100)
        }
        model = train_transformer(x_train_num, x_train_cat, survival_train, time_grid, cat_cardinalities, params)
    elif model_type == "deephit_transformer":
        params = {
            'hidden_size': trial.suggest_categorical("hidden_size", [8, 16, 32]),
            'num_attention_heads': trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
            'num_layers': trial.suggest_int("num_layers", 2, 4),
            'dim_feedforward': trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
            'dropout': trial.suggest_float("dropout", 0.0, 0.5),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float("weight_decay", 0.0, 0.3),
            'epochs': trial.suggest_int("epochs", 10, 100),
            'alpha': trial.suggest_float("alpha", 0.1, 1.0),
            'sigma': trial.suggest_float("sigma", 0.1, 1.0)
        }
        model = train_deephit_transformer(x_train_num, x_train_cat, survival_train, time_grid, cat_cardinalities, params, NUM_DURATIONS)

    elif model_type == "deepsurv":
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            'num_nodes1': trial.suggest_int("num_nodes1", 32, 128),
            'num_nodes2': trial.suggest_int("num_nodes2", 32, 128),
            'dropout': trial.suggest_float("dropout", 0.0, 0.5),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'epochs': trial.suggest_int("epochs", 10, 100)
        }
        model = train_deepsurv(x_train_num, x_train_cat, survival_train, cat_cardinalities, params)

    elif model_type == "deephit":
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            'num_nodes': trial.suggest_int("num_nodes", 32, 128),
            'dropout': trial.suggest_float("dropout", 0.1, 0.5),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'alpha': trial.suggest_float("alpha", 0.1, 1.0),
            'sigma': trial.suggest_float("sigma", 0.1, 1.0),
            'epochs': trial.suggest_int("epochs", 10, 100)
        }
        model = train_deephit(x_train_num, x_train_cat, survival_train, cat_cardinalities, params, NUM_DURATIONS)

    wrapper = SurvivalModelWrapper(model, model_type, time_grid)
    surv_df = wrapper.predict_surv_df((x_val_num, x_val_cat))
    from sksurv.metrics import integrated_brier_score
    ibs = integrated_brier_score(et_val, et_val, surv_df.values, time_grid)
    
    return ibs

def train_and_evaluate_models_tokenized(model_types, best_params, x_train_num, x_train_cat, x_test_num, x_test_cat,
                              survival_train, survival_test, cat_cardinalities, times, n_runs=5):
    et_train = Surv.from_arrays(survival_train['event'], survival_train['time'])
    et_test = Surv.from_arrays(survival_test['event'], survival_test['time'])
    trained_models = {}
    results = {model: {'C-Index': [], 'Integrated Brier Score': []} for model in model_types}
    for h in times:
        for model in model_types:
            results[model][f"Brier@{h}"] = []
            results[model][f"AUC@{h}"] = []
            results[model][f"Concordance-IPCW@{h}"] = []

    for run in range(n_runs):
        wandb.log({"Run": run + 1})
        for model_type in model_types:
            if model_type == "transformer":
                model = train_transformer(x_train_num, x_train_cat, survival_train, times, cat_cardinalities, best_params[model_type])
            elif model_type == "deephit_transformer":
                model = train_deephit_transformer(x_train_num, x_train_cat, survival_train, times, cat_cardinalities, best_params[model_type], num_durations=len(times) + 1)
            elif model_type == "deepsurv":
                model = train_deepsurv(x_train_num, x_train_cat, survival_train, cat_cardinalities, best_params[model_type])
            elif model_type == "deephit":
                model = train_deephit(x_train_num, x_train_cat, survival_train, cat_cardinalities, best_params[model_type], NUM_DURATIONS=len(times) + 1)
            else:
                raise ValueError(f"Unsupported model: {model_type}")

            wrapper = SurvivalModelWrapper(model, model_type, times)
            surv_df = wrapper.predict_surv_df((x_test_num, x_test_cat))
            risk_df = 1 - surv_df.values

            final_risk = risk_df[:, -1]
            results[model_type]['C-Index'].append(concordance_index_censored(et_test['event'], et_test['time'], final_risk)[0])

            brs = brier_score(et_test, et_test, surv_df.values, times)[1]
            ibs = integrated_brier_score(et_test, et_test, surv_df.values, times)
            results[model_type]['Integrated Brier Score'].append(ibs)

            try:
                aucs, mean_auc = cumulative_dynamic_auc(et_test, et_test, final_risk, times)[0]
            except Exception:
                aucs = [0] * len(times)
            HORIZONS = [0.25, 0.5, 0.75]
            for i, h in enumerate(times):
                results[model_type][f"Brier@{h}"].append(brs[i])
                results[model_type][f"AUC@{h}"].append(aucs[i])
                results[model_type][f"Concordance-IPCW@{h}"].append(
                    concordance_index_ipcw(et_test, et_test, risk_df[:, i], tau=times[i])[0]
                )

    summary_results = {}
    for model in model_types:
        summary = {}
        for k, v in results[model].items():
            summary[f"{k} Mean"] = np.mean(v)
            summary[f"{k} Std"] = np.std(v)
        wandb.log(summary)
        summary_results[model] = summary

    return summary_results