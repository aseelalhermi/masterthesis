from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
import pandas as pd

def get_survival_probs_manual(surv_df, times):
    """
    Interpolates S(t | x) at specific time points using numpy, per column.
    
    Parameters:
        surv_df (pd.DataFrame): [time, n_samples]
        times (array-like): list of times to interpolate S(t) at
    
    Returns:
        pd.DataFrame: [n_samples, len(times)] with survival probabilities
    """
    times = np.asarray(times)
    time_grid = surv_df.index.values
    surv_values = surv_df.values  # shape: [n_times, n_samples]
    
    # Interpolate for each sample (column)
    results = {
        f"S(t={t})": np.interp(t, time_grid, surv_values[:, i])
        for i in range(surv_values.shape[1])
        for t in times
    }

    # Reshape and return as DataFrame [n_samples, len(times)]
    result_df = pd.DataFrame(
        np.stack([
            [np.interp(t, time_grid, surv_values[:, i]) for t in times]
            for i in range(surv_values.shape[1])
        ]),
        columns=[f"S(t={t})" for t in times]
    )
    return result_df

class SurvivalModelWrapper:
    def __init__(self, model, model_type, time_grid):
        self.model = model
        self.model_type = model_type
        self.times = time_grid

    def predict_surv_df(self, x):
        """
        Returns survival probability matrix [n_samples, len(times)] as a DataFrame.
        Supports sksurv, PyCox, DeepHit, custom Transformer models.
        """
        if self.model_type in ['cox', 'rsf', 'gb']:
            surv_funcs = self.model.predict_survival_function(x)
            surv_probs = np.asarray([fn(self.times) for fn in surv_funcs])

            return pd.DataFrame(surv_probs, columns=self.times)
        elif self.model_type in ['transformer', 'deepsurv', 'deephit']:
            # Predict survival functions: shape = [n_times_in_training, n_samples]
            surv_df = self.model.predict_surv_df(x)
            
            surv_scores = get_survival_probs_manual(surv_df, self.times)
        
            # Return transposed version so it's shape: [n_samples, n_times]
            return surv_scores


        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
def get_survival_quantiles(surv_df, quantiles=[0.5]):
    """
    Compute survival quantiles (e.g. median survival) from a survival dataframe.
    
    Parameters:
    - surv_df: pd.DataFrame of shape [time, n_samples]
    - quantiles: list of quantiles (e.g., [0.5] for median)
    
    Returns:
    - pd.DataFrame of shape [n_samples, len(quantiles)], with time estimates for each quantile
    """
    results = {}
    for q in quantiles:
        target_surv = 1 - q  # e.g. 0.5 for median → S(t) = 0.5
        # For each sample (column), find first time where S(t) <= target
        times = surv_df.index.values
        qt = surv_df.apply(lambda s: np.interp(target_surv, s[::-1], times[::-1]), axis=0)
        results[f'q{int(q*100)}'] = qt.values

    return pd.DataFrame(results, index=surv_df.columns)

def tune_model(trial, model_type, x_train, survival_train, x_val, survival_val, time_grid):
    from sksurv.util import Surv

    et_train = Surv.from_arrays(survival_train["event"], survival_train["time"])
    et_val = Surv.from_arrays(survival_val["event"], survival_val["time"])

    NUM_DURATIONS = 100
    HORIZONS = [0.25, 0.5, 0.75]
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)

    if model_type == "cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        model = CoxPHSurvivalAnalysis(alpha=alpha)

    elif model_type == "rsf":
        from sksurv.ensemble import RandomSurvivalForest
        model = RandomSurvivalForest(
            n_estimators=trial.suggest_int('n_estimators', 10, 200),
            max_depth=trial.suggest_int('max_depth', 2, 32),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 10, 100),
            max_samples=trial.suggest_float('max_samples', 0.1, 1.0),
            random_state=42
        )

    elif model_type == "gb":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=trial.suggest_int('n_estimators', 10, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 32),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            random_state=42
        )


    model.fit(x_train, et_train)
    event_times = survival_train["time"][survival_train["event"] == 1]

    # Include min, specified quantiles, and max
    HORIZONS = [0.25, 0.5, 0.75]

    # Evaluation
    wrapper = SurvivalModelWrapper(model, model_type, time_grid)
    
    surv_df = wrapper.predict_surv_df(x_val)

    from sksurv.metrics import integrated_brier_score
    ibs = integrated_brier_score(et_val, et_val, surv_df.values, time_grid)
    
    return ibs

from sklearn.model_selection import StratifiedKFold

def tune_model_cv(trial, model_type, x, durations, events, time_grid, n_splits=5):
    from sksurv.util import Surv
    from sksurv.metrics import integrated_brier_score
    from sklearn.model_selection import StratifiedKFold


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(x, events):
        x_train, x_val = x[train_idx], x[val_idx]
        d_train, d_val = durations[train_idx], durations[val_idx]
        e_train, e_val = events[train_idx], events[val_idx]
        surv_train = Surv.from_arrays(e_train, d_train)
        surv_val = Surv.from_arrays(e_val, d_val)

        score = tune_model(
            trial=trial,
            model_type=model_type,
            x_train=x_train,
            survival_train={"time": d_train, "event": e_train},
            x_val=x_val,
            survival_val={"time": d_val, "event": e_val},
            time_grid=time_grid
        )
        scores.append(score)

    return np.mean(scores)


def train_and_evaluate_models(model_types, best_params, x_train, x_test, survival_train, survival_test, n_runs, times):
    from sksurv.util import Surv
    from sksurv.metrics import (
        concordance_index_censored,
        concordance_index_ipcw,
        brier_score,
        cumulative_dynamic_auc
    )
    import numpy as np
    import wandb

    HORIZONS = [0.25, 0.5, 0.75]
    max_time = survival_train["time"].max()

    event_times = survival_train["time"][survival_train["event"] == 1]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    trained_models = {}
    results = {
        model_type: {"C-Index": []} for model_type in model_types
    }
    for h in times:
        for model_type in model_types:
            results[model_type][f"Brier@{h}"] = []
            results[model_type][f"AUC@{h}"] = []
            results[model_type][f"Concordance-IPCW@{h}"] = []

    et_train = Surv.from_arrays(survival_train["event"], survival_train["time"])
    et_test = Surv.from_arrays(survival_test["event"], survival_test["time"])

    for run in range(n_runs):
        wandb.log({"Run": run + 1})
        for model_type in model_types:
            if model_type == "cox":
                model = CoxPHSurvivalAnalysis(**best_params[model_type])
            elif model_type == "rsf":
                model = RandomSurvivalForest(**best_params[model_type], random_state=run)
            elif model_type == "gb":
                model = GradientBoostingSurvivalAnalysis(**best_params[model_type], random_state=run)

            model.fit(x_train, et_train)

            NUM_DURATIONS = 3
            HORIZONS = [0.25, 0.5, 0.75]

            event_times = survival_test["time"][survival_test["event"] == 1]

            # Wrap model
            wrapper = SurvivalModelWrapper(model, model_type, times)
            surv_df = wrapper.predict_surv_df(x_test)  # [n_samples, n_times]
            risk_df = 1 - surv_df.values

            # C-index (censored)
            final_risk = risk_df[:, -1]
            c_index = concordance_index_censored(et_test["event"], et_test["time"], final_risk)[0]
            results[model_type]["C-Index"].append(c_index)

            # Brier and AUC
            brs = brier_score(et_test, et_test, surv_df.values, times)[1]
            
            from sksurv.metrics import integrated_brier_score
            ibs = integrated_brier_score(et_test, et_test, surv_df.values, times)
            results[model_type].setdefault("Integrated Brier Score", []).append(ibs)

            from lifelines import KaplanMeierFitter

            def filter_ipcw_safe_times(times, df):
                censoring = 1 - df["event"]
                kmf = KaplanMeierFitter()
                kmf.fit(df["time"], event_observed=censoring)
                surv_probs = kmf.survival_function_at_times(times).values
                return [t for t, p in zip(times, surv_probs) if p > 0]

            try:
                aucs = cumulative_dynamic_auc(et_test, et_test, final_risk, times)[0]
            except ValueError as e:
                print(f"[AUC Warning] {e} — attempting to filter unsafe time points.")
                safe_times = filter_ipcw_safe_times(times, survival_test)
                if len(safe_times) == 0:
                    print("No valid time points remain after filtering. Skipping AUC.")
                    aucs = [0] * len(times)
                else:
                  try:
                    aucs = cumulative_dynamic_auc(et_test, et_test, final_risk, safe_times)[0]

                  except ValueError as e:
                    print("No valid time points remain after filtering. Skipping AUC.")
                    aucs = [0] * len(times)

            for i, h in enumerate(times):
                results[model_type][f"Brier@{h}"].append(brs[i])
                results[model_type][f"AUC@{h}"].append(aucs[i])
                results[model_type][f"Concordance-IPCW@{h}"].append(
                    concordance_index_ipcw(et_test, et_test, risk_df[:, i], tau=times[i])[0]
                )

    summary_results = {}
    for model_type in model_types:
        summary = {}
        for k, v in results[model_type].items():
            summary[f"{k} Mean"] = np.mean(v)
            summary[f"{k} Std"] = np.std(v)
        wandb.log(summary)
        summary_results[model_type] = summary

    return trained_models, summary_results



