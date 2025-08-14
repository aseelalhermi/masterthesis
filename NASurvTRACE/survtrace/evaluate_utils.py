from collections import defaultdict
from sksurv.metrics import (
    concordance_index_ipcw, brier_score,
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc
)
import pdb
from sksurv.util import Surv
from matplotlib import pyplot as plt
import numpy as np
import wandb
from lifelines import KaplanMeierFitter


class Evaluator:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.df_train_all = df.loc[train_index]
        
    def eval_single(self, model, test_set, val_batch_size=None):
        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = get_target(df_train_all)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype=[('e', bool), ('t', float)])
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']

        df_test, df_y_test, attention_mask_test = test_set
        surv = model.predict_surv(df_test, batch_size=val_batch_size, attention_mask=attention_mask_test)
        risk = 1 - surv

        durations_test, events_test = get_target(df_y_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                          dtype=[('e', bool), ('t', float)])

        metric_dict = defaultdict(list)

        # Per-time metrics
        try:
            brs = brier_score(et_test, et_test, surv.to("cpu").numpy()[:, 1:-1], times)[1]
            for i, horizon in enumerate(horizons):
                metric_dict[f"{horizon}_brier"] = brs[i]
        except Exception as e:
            print("Brier Score failed:", e)

        try:
            cis = []
            for i, _ in enumerate(times):
                c = concordance_index_ipcw(
                    et_test, et_test, estimate=risk[:, i + 1].to("cpu").numpy(), tau=times[i]
                )[0]
                metric_dict[f"{horizons[i]}_ipcw"] = c
        except Exception as e:
            print("IPCW C-index failed:", e)

        # Additional metrics
        try:
            # Censored (Harrell) Concordance Index
            c_cens, _, _, _, _ = concordance_index_censored(events_test.astype(bool), durations_test, risk.mean(axis=1).cpu().numpy())
            
            metric_dict["c_index_censored"] = c_cens
        except Exception as e:
            print("Censored C-index failed:", e)

        try:
            surv_fn = Surv.from_arrays(events_train.astype(bool), durations_train)
            test_fn = Surv.from_arrays(events_test.astype(bool), durations_test)

            ibs = integrated_brier_score(
                test_fn, test_fn, surv.to("cpu").numpy()[:, 1:-1], times
            )
            metric_dict["integrated_brier"] = ibs
        except Exception as e:
            print("Integrated Brier Score failed:", e)

        try:
            surv_fn = Surv.from_arrays(events_train.astype(bool), durations_train)
            test_fn = Surv.from_arrays(events_test.astype(bool), durations_test)

            risk_scores = risk.to("cpu").numpy()[:, 1:-1]
            valid_times = times[:np.sum(np.isfinite(durations_test)) - 1]

            # Remove time points where the censoring survival function becomes 0
            aucs, mean_auc = cumulative_dynamic_auc(
                test_fn, test_fn, risk_scores, valid_times
            )
            metric_dict["cumulative_auc"] = mean_auc
            for i, h in enumerate(horizons):
                metric_dict[f"{h}_auc"] = aucs[i]

        except ValueError as e:
            print("AUC failed:", e)

        for k, v in metric_dict.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
        return metric_dict

    def eval_multi(self, model, test_set, val_batch_size=10000):
        times = model.config['duration_index'][1:-1]
        horizons = model.config['horizons']
        df_train_all = self.df_train_all
        get_target = lambda df, risk: (df['duration'].values, df['event_{}'.format(risk)].values)
        df_test, df_y_test = test_set

        metric_dict = defaultdict(list)
        for risk_idx in range(model.config.num_event):
            durations_train, events_train = get_target(df_train_all, risk_idx)
            durations_test, events_test = get_target(df_y_test, risk_idx)
            
            surv = model.predict_surv(df_test, batch_size=val_batch_size, event=risk_idx)
            risk = 1 - surv

            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])

            brs = brier_score(et_train, et_test, surv.to("cpu").numpy()[:,1:-1], times)[1]
            cis = []
            for i, _ in enumerate(times):
                cis.append(concordance_index_ipcw(et_train, et_test, risk[:, i+1].to("cpu").numpy(), times[i])[0])            
                metric_dict[f'{horizons[i]}_ipcw_{risk_idx}'] = cis[i]
                metric_dict[f'{horizons[i]}_brier_{risk_idx}'] = brs[i]

            for horizon in enumerate(horizons):
                print("Event: {} For {} quantile,".format(risk_idx,horizon[1]))
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
                print("Brier Score:", brs[horizon[0]])
        
        return metric_dict

    def eval(self, model, test_set, confidence=None, val_batch_size=None):
        '''do evaluation.
        if confidence is not None, it should be in (0, 1) and the confidence
        interval will be given by bootstrapping.
        '''
        print("***"*10)
        print("start evaluation")
        print("***"*10)

        if confidence is None:
            if model.config['num_event'] > 1:
                return self.eval_multi(model, test_set, val_batch_size)
            else:
                return self.eval_single(model, test_set, val_batch_size)

        else:
            # do bootstrapping
            stats_dict = defaultdict(list)
            for i in range(10):
                df_test = test_set[0].sample(test_set[0].shape[0], replace=True)
                df_y_test = test_set[1].loc[df_test.index]
                
                if model.config['num_event'] > 1:
                    res_dict = self.eval_multi(model, (df_test, df_y_test), val_batch_size)
                else:
                    res_dict = self.eval_single(model, (df_test, df_y_test), val_batch_size)

                for k in res_dict.keys():
                    stats_dict[k].append(res_dict[k])

            metric_dict = {}
            # compute confidence interveal 95%
            alpha = confidence
            p1 = ((1-alpha)/2) * 100
            p2 = (alpha+((1.0-alpha)/2.0)) * 100
            for k in stats_dict.keys():
                stats = stats_dict[k]
                lower = max(0, np.percentile(stats, p1))
                upper = min(1.0, np.percentile(stats, p2))
                # print(f'{alpha} confidence interval {lower} and {upper}')
                print(f'{alpha} confidence {k} average:', (upper+lower)/2)
                print(f'{alpha} confidence {k} interval:', (upper-lower)/2)
                metric_dict[k] = [(upper+lower)/2, (upper-lower)/2]

            return metric_dict

def plot_censoring_by_risk_group(durations, events, risk_scores, times, name="censoring_distribution"):

    quantiles = np.quantile(risk_scores, [0.33, 0.66])
    group_labels = np.digitize(risk_scores, quantiles)
    group_names = ["Low Risk", "Medium Risk", "High Risk"]

    fig, ax = plt.subplots()
    for group in range(3):
        group_mask = group_labels == group
        group_durations = durations[group_mask]
        group_events = events[group_mask]

        ax.hist(group_durations[group_events == 0], bins=20, alpha=0.5, label=f"{group_names[group]} (censored)")

    ax.set_title("Censoring Distribution by Risk Group")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.legend()
    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)

def plot_survival_analysis(surv, risk, durations_test, events_test, times, horizons, metrics):
  # === 1. Avg Survival ===
  try:
      avg_surv = surv.to("cpu").numpy().mean(axis=0)[1:-1]
      fig, ax = plt.subplots()
      ax.plot(times, avg_surv)
      ax.set_title("Average Survival Curve")
      ax.set_xlabel("Time")
      ax.set_ylabel("Survival Probability")
      wandb.log({"avg_survival_curve": wandb.Image(fig)})
      plt.close(fig)
  except Exception as e:
      print("Avg survival curve failed:", e)

  # === 2. Time-dependent AUC ===
  try:
      if "cumulative_auc" in metrics:
          aucs = [metrics[f"{h}_auc"] for h in horizons if f"{h}_auc" in metrics]
          fig, ax = plt.subplots()
          ax.plot(times, aucs)
          ax.set_title("Time-dependent AUC")
          ax.set_xlabel("Time")
          ax.set_ylabel("AUC")
          wandb.log({"time_auc_curve": wandb.Image(fig)})
          plt.close(fig)
  except Exception as e:
      print("AUC curve failed:", e)

  # === 3. Brier Score ===
  try:
      if f"{horizons[0]}_brier" in metrics:
          brs = [metrics[f"{h}_brier"] for h in horizons if f"{h}_brier" in metrics]
          fig, ax = plt.subplots()
          ax.plot(times, brs)
          ax.set_title("Brier Score Over Time")
          ax.set_xlabel("Time")
          ax.set_ylabel("Brier Score")
          wandb.log({"brier_score_curve": wandb.Image(fig)})
          plt.close(fig)
  except Exception as e:
      print("Brier score plot failed:", e)

  # === 4. KM vs Model ===
  try:
      kmf = KaplanMeierFitter()
      kmf.fit(durations=durations_test, event_observed=events_test)
      fig, ax = plt.subplots()
      ax.step(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], label="KM", where="post")
      ax.plot(times, avg_surv, label="Model", linestyle="--")
      ax.set_title("KM vs Model Survival")
      ax.set_xlabel("Time")
      ax.set_ylabel("Survival Probability")
      ax.legend()
      wandb.log({"km_vs_model_survival": wandb.Image(fig)})
      plt.close(fig)
  except Exception as e:
      print("KM vs Model failed:", e)

  # === 5. CI of Model Survival ===
  try:
      surv_np = surv.to("cpu").numpy()[:, 1:-1]
      avg_surv = surv_np.mean(axis=0)
      std_surv = surv_np.std(axis=0)
      upper = np.minimum(avg_surv + 1.96 * std_surv / np.sqrt(surv_np.shape[0]), 1.0)
      lower = np.maximum(avg_surv - 1.96 * std_surv / np.sqrt(surv_np.shape[0]), 0.0)
      fig, ax = plt.subplots()
      ax.plot(times, avg_surv, label="Avg Survival")
      ax.fill_between(times, lower, upper, alpha=0.3, label="95% CI")
      ax.legend()
      ax.set_title("Model Survival Curve with CI")
      wandb.log({"model_survival_with_ci": wandb.Image(fig)})
      plt.close(fig)
  except Exception as e:
      print("CI plot failed:", e)

  # === 6. Risk Group Survival Curves ===
  try:
      risk_scores = risk.mean(axis=1).detach().cpu().numpy()
      quantiles = np.quantile(risk_scores, [0.33, 0.66])
      group_labels = np.digitize(risk_scores, quantiles)
      surv_np = surv.to("cpu").numpy()[:, 1:-1]
      fig, ax = plt.subplots()
      for group in range(3):
          group_mask = group_labels == group
          ax.plot(times, surv_np[group_mask].mean(axis=0), label=["Low", "Medium", "High"][group])
      ax.legend()
      ax.set_title("Survival by Risk Group")
      wandb.log({"survival_by_risk_group": wandb.Image(fig)})
      plt.close(fig)
  except Exception as e:
      print("Survival by risk group failed:", e)

  # === 7. Censoring Histogram by Risk ===
  try:
      plot_censoring_by_risk_group(durations_test, events_test, risk_scores, times)
  except Exception as e:
      print("Censoring histogram failed:", e)

