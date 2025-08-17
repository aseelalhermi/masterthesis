from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from augmentations import embed_data_mask
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    brier_score,
    cumulative_dynamic_auc,
    integrated_brier_score
)
from sksurv.util import Surv
import wandb
from lifelines import KaplanMeierFitter

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse

def mean_abs_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts[:, 0].squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_absolute_error(y_test.cpu(), y_pred.cpu())
        return rmse


def compute_baseline_cumulative_hazard(durations, events, log_hazards):
    durations = np.asarray(durations)
    events = np.asarray(events)
    hazards = np.exp(log_hazards)

    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    hazards = hazards[order]

    unique_times, _ = np.unique(durations[events == 1], return_counts=True)
    cum_baseline_hazard = []
    ch = 0.0

    for t in unique_times:
        risk_set = durations >= t
        denom = hazards[risk_set].sum()
        d_i = (durations == t) & (events == 1)
        n_i = d_i.sum()

        if denom > 0:
            ch += n_i / denom
        cum_baseline_hazard.append(ch)

    return np.array(unique_times), np.array(cum_baseline_hazard)

def get_survival_probs_manual(surv_df, times):
    times = np.asarray(times)
    time_grid = surv_df.index.values
    surv_values = surv_df.values  # [n_times, n_samples]
    result_df = pd.DataFrame(
        np.stack([
            [np.interp(t, time_grid, surv_values[:, i]) for t in times]
            for i in range(surv_values.shape[1])
        ]),
        columns=[f"S(t={t:.2f})" for t in times]
    )
    return result_df

def ibs(dataset_name, model, dloader, device, vision_dset, time_grid, horizons=[0.25, 0.5, 0.75], log_to_wandb=True, decoder=False, decoder_model=None):
    model.eval()
    hazards_list, durations_list, events_list = [], [], []

    with torch.no_grad():
        for data in dloader:
            x_categ, x_cont, y_gts, cat_mask, con_mask = (
                data[0].to(device), data[1].to(device), data[2].to(device),
                data[3].to(device), data[4].to(device)
            )
            
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            log_hazards = model.mlpfory(y_reps).squeeze(1)
            hazards_list.append(torch.exp(log_hazards).cpu().numpy())
            durations_list.append(y_gts[:, 0].cpu().numpy())
            events_list.append(y_gts[:, 1].cpu().numpy())

    hazards = np.concatenate(hazards_list)
    durations = np.concatenate(durations_list)
    events = np.concatenate(events_list)
    et = Surv.from_arrays(events.astype(bool), durations)

    # Compute Breslow baseline
    unique_times, cum_base_hazard = compute_baseline_cumulative_hazard(durations, events, np.log(hazards))
    if len(unique_times) == 0 or len(cum_base_hazard) == 0:
      print("No events in dataset. Skipping survival computation.")
      return None
    surv = np.exp(-np.outer(hazards, cum_base_hazard))
    surv_df = pd.DataFrame(surv.T, index=unique_times)

    # Use full time grid for survival prediction
    full_times = np.sort(np.unique(durations))
    full_surv_interp = get_survival_probs_manual(surv_df, full_times)

    # Select quantile-based horizons for metric evaluation
    event_times = durations[events == 1]

    horizon_surv = get_survival_probs_manual(surv_df, time_grid)

    if np.sum(events) == 0 or np.isnan(horizon_surv.values[:, 0]).sum()!=0:
        print("Warning: all samples are censored. IPCW-based metrics will be skipped.")
        ibs = np.nan
    else:
        ibs = integrated_brier_score(et, et, horizon_surv.values, time_grid)  
    return ibs

def metrics_surv(dataset_name, model, dloader, device, vision_dset, time_grid, horizons=[0.25, 0.5, 0.75], log_to_wandb=True, decoder=False, decoder_model=None):
    model.eval()

    hazards_list, durations_list, events_list = [], [], []

    with torch.no_grad():
      for data in dloader:
          x_categ, x_cont, y_gts, cat_mask, con_mask = (
              data[0].to(device), data[1].to(device), data[2].to(device),
              data[3].to(device), data[4].to(device)
          )
          _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
          reps = model.transformer(x_categ_enc, x_cont_enc)
          y_reps = reps[:, 0, :]
          log_hazards = model.mlpfory(y_reps).squeeze(1)

          hazards_list.append(torch.exp(log_hazards).cpu().numpy())

          durations_list.append(y_gts[:, 0].cpu().numpy())
          events_list.append(y_gts[:, 1].cpu().numpy())

    hazards = np.concatenate(hazards_list)
    durations = np.concatenate(durations_list)
    events = np.concatenate(events_list)
    et = Surv.from_arrays(events.astype(bool), durations)

    # Compute Breslow baseline
    unique_times, cum_base_hazard = compute_baseline_cumulative_hazard(durations, events, np.log(hazards))
    surv = np.exp(-np.outer(hazards, cum_base_hazard))
    surv_df = pd.DataFrame(surv.T, index=unique_times)

    # Use full time grid for survival prediction
    full_times = np.sort(np.unique(durations))
    full_surv_interp = get_survival_probs_manual(surv_df, full_times)

    # Select quantile-based horizons for metric evaluation
    event_times = durations[events == 1]

    horizon_surv = get_survival_probs_manual(surv_df, time_grid)


    if np.sum(events) == 0 or np.isnan(horizon_surv.values[:, 0]).sum()!=0:
      print("Warning: all samples are censored. IPCW-based metrics will be skipped.")
      ipcw_scores = [np.nan for _ in time_grid]
      ipcw_025 = np.nan
    else:
      ipcw_scores = [
          concordance_index_ipcw(et, et, 1 - horizon_surv.values[:, i], tau=t)[0]
          for i, t in enumerate(time_grid)
      ]

    # Brier Score
    brier_scores = brier_score(et, et, horizon_surv.values, time_grid)[1]

    def filter_ipcw_safe_times(times, df):
      censoring = 1 - df["event"]
      kmf = KaplanMeierFitter()
      kmf.fit(df["time"], event_observed=censoring)
      surv_probs = kmf.survival_function_at_times(times).values
      return [t for t, p in zip(times, surv_probs) if p > 0]

    try:
      auc_scores = cumulative_dynamic_auc(et, et, 1 - horizon_surv.values, time_grid)[0]
    except ValueError as e:
      print(f"[AUC Warning] {e} — attempting to filter unsafe time points.")
      safe_times = filter_ipcw_safe_times(time_grid, et)
      if len(safe_times) == 0:
          print("No valid time points remain after filtering. Skipping AUC.")
          auc_scores = [0] * len(time_grid)
      else:
          try:
              auc_scores = cumulative_dynamic_auc(et, et, 1 - horizon_surv.values, time_grid)[0]
          except ValueError as e:
              print("No valid time points remain after filtering. Skipping AUC.")
              auc_scores = [0] * len(time_grid)

    censored_c_index = concordance_index_censored(events.astype(bool), durations, hazards)[0]

    ibs = integrated_brier_score(et, et, horizon_surv.values, time_grid)


    metrics = {
      "C-Index": censored_c_index,
      "Integrated Brier Score": ibs,
      **{f"Brier@{round(t, 2)}": b for t, b in zip(time_grid, brier_scores)},
      **{f"AUC@{round(t, 2)}": a for t, a in zip(time_grid, auc_scores)},
      **{f"Concordance-IPCW@{round(t, 2)}": c for t, c in zip(time_grid, ipcw_scores)}
    }

    if log_to_wandb:
      wandb.log(metrics)

    return metrics

def metrics_surv_saintwithtime(model, dloader, device, time_grid, vision_dset, log_to_wandb=True):
    risk_list, durations_list, events_list = [], [], []
    surv_probs_list = []

    with torch.no_grad():
        for data in dloader:
            x_categ, x_cont, y_gts, cat_mask, con_mask = (
                data[0].to(device), data[1].to(device), data[2].to(device),
                data[3].to(device), data[4].to(device)
            )
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model.base_model, vision_dset
            )
            batch_size = x_categ.shape[0]
            time_ids = torch.tensor(time_grid, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
            raw_output = model(x_categ_enc, x_cont_enc, time_ids)
            hazard_increments = F.softplus(raw_output)
            cumulative_hazard = torch.cumsum(hazard_increments, dim=1)
            surv_probs = torch.exp(-cumulative_hazard).clamp(min=0.0, max=1.0)
            risk_scores = cumulative_hazard[:, -1]
            
            risk_list.append(risk_scores.cpu().numpy())
            surv_probs_list.append(surv_probs.cpu().numpy())
            durations_list.append(y_gts[:, 0].cpu().numpy())
            events_list.append(y_gts[:, 1].cpu().numpy())

    risks = np.concatenate(risk_list)
    surv_probs_all = np.concatenate(surv_probs_list, axis=0)  # (N, len(time_grid))
    durations = np.concatenate(durations_list)
    events = np.concatenate(events_list)
    et = Surv.from_arrays(events.astype(bool), durations)

    # === Metrics directly from survival probs ===

    # IPCW concordance & AUC
    if np.sum(events) == 0 or np.isnan(surv_probs_all).any():
        print("Warning: all samples are censored or NaNs present. IPCW-based metrics skipped.")
        ipcw_scores = [np.nan for _ in time_grid]
        auc_scores = [np.nan for _ in time_grid]
    else:
        ipcw_scores = [
            concordance_index_ipcw(et, et, 1 - surv_probs_all[:, i], tau=t)[0]
            for i, t in enumerate(time_grid)
        ]
        try:
            auc_scores = cumulative_dynamic_auc(et, et, 1 - surv_probs_all, time_grid)[0]
        except ValueError:
            def filter_ipcw_safe_times(times, et):
                censoring = 1 - et["event"]
                kmf = KaplanMeierFitter()
                kmf.fit(et["time"], event_observed=censoring)
                return [t for t, p in zip(times, kmf.survival_function_at_times(times).values) if p > 0]
            safe_times = filter_ipcw_safe_times(time_grid, et)
            auc_scores = [0.0] * len(safe_times) if not safe_times else cumulative_dynamic_auc(et, et, 1 - surv_probs_all, safe_times)[0]

    # Brier score & Integrated Brier Score
    brier_scores = brier_score(et, et, surv_probs_all, time_grid)[1]
    censored_c_index = concordance_index_censored(events.astype(bool), durations, risks)[0]
    ibs = integrated_brier_score(et, et, surv_probs_all, time_grid)

    metrics = {
        "C-Index": censored_c_index,
        "Integrated Brier Score": ibs,
        **{f"Brier@{round(t, 2)}": b for t, b in zip(time_grid, brier_scores)},
        **{f"AUC@{round(t, 2)}": a for t, a in zip(time_grid, auc_scores)},
        **{f"Concordance-IPCW@{round(t, 2)}": c for t, c in zip(time_grid, ipcw_scores)}
    }

    if log_to_wandb:
        import wandb
        wandb.log(metrics)

    return metrics

class SAINTDeepHitPredictor:
    def __init__(self, model, duration_index, device, num_durations=5):
        self.model = model
        self.duration_index = duration_index
        self.device = device
        self.num_durations = num_durations
        self.model.eval()

    def predict_pmf(self, dloader):
        """
        Predict Probability Mass Function (PMF) using softmax over SAINT output.
        """
        pmf_all = []

        with torch.no_grad():
            for data in dloader:
                x_categ, x_cont, _,_,_,_, cat_mask, con_mask = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    data[3].to(self.device),
                    data[4].to(self.device),
                    data[5].to(self.device),
                    data[6].to(self.device),
                    data[7].to(self.device)
                )
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model, vision_dset=False)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                phi = self.model.mlpfory(y_reps)

                # Convert logits to PMF using padded softmax
                phi_padded = F.pad(phi, (1, 0))  # Pad zeros at the start
                pmf = F.softmax(phi_padded, dim=1)[:, 1:]
                pmf_all.append(pmf.cpu())

        return torch.cat(pmf_all, dim=0)  # [n_samples, num_durations]

    def predict_surv_df(self, dloader):
        """
        Return survival function estimates as a DataFrame [time, n_samples].
        """
        pmf = self.predict_pmf(dloader)
        cif = torch.cumsum(pmf, dim=1)
        surv = 1.0 - cif
        return pd.DataFrame(surv.numpy().T, index=self.duration_index[1:])  # exclude first (0) and last (max) for midpoint grid

    def evaluate(self, dloader, durations, events, time_grid, log_to_wandb=False):
      """
      Evaluate C-index, Brier Score, AUC, IPCW using sksurv and manual logic (like metrics_surv).
      """

      # Get PMF predictions and compute survival curves
      pmf = self.predict_pmf(dloader)
      cif = torch.cumsum(pmf, dim=1)
      surv = 1.0 - cif
      surv_df = pd.DataFrame(surv.numpy().T, index=self.duration_index[1:])  # Shape: [time, n]

      # Prepare ground truth
      durations = np.asarray(durations)
      events = np.asarray(events)
      et = Surv.from_arrays(events.astype(bool), durations)

      # Interpolate for exact time_grid points
      horizon_surv = get_survival_probs_manual(surv_df, time_grid)


      if np.sum(events) == 0 or np.isnan(horizon_surv.values[:, 0]).sum() != 0:
          print("Warning: all samples are censored. IPCW-based metrics will be skipped.")
          ipcw_scores = [np.nan for _ in time_grid]
      else:
          ipcw_scores = [
              concordance_index_ipcw(et, et, 1 - horizon_surv.values[:, i], tau=t)[0]
              for i, t in enumerate(time_grid)
          ]

      try:
          auc_scores = cumulative_dynamic_auc(et, et, 1 - horizon_surv.values, time_grid)[0]
      except ValueError:
          print("AUC computation failed. Filling with zeros.")
          auc_scores = [0] * len(time_grid)

      brier_scores = brier_score(et, et, horizon_surv.values, time_grid)[1]
      ibs = integrated_brier_score(et, et, horizon_surv.values, time_grid)
      censored_c_index = concordance_index_censored(events.astype(bool), durations, 1 - horizon_surv.values[:, 0])[0]

      metrics = {
          "C-Index": censored_c_index,
          "Integrated Brier Score": ibs,
          **{f"Brier@{round(t, 2)}": b for t, b in zip(time_grid, brier_scores)},
          **{f"AUC@{round(t, 2)}": a for t, a in zip(time_grid, auc_scores)},
          **{f"Concordance-IPCW@{round(t, 2)}": c for t, c in zip(time_grid, ipcw_scores)}
      }

      if log_to_wandb:
          import wandb
          wandb.log(metrics)

      return metrics
