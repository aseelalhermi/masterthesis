import argparse
import numpy as np
import optuna
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import SAINT
from data_openml import data_prep_openml, DataSetCatCon
from utils import count_parameters, mean_abs_error, ibs
from utils import SAINTDeepHitPredictor
from models.timemodel import SAINTWithTime
from datasets import DeepHitDataset, preprocess_survival_dataset
from losses import CoxPHLoss, pair_rank_mat, DeepHitSingleLoss, CombinedLossMonotonic
from augmentations import embed_data_mask


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--use_openml', action='store_true', help="Whether to use OpenML datasets.")
parser.add_argument('--survival', action='store_true', help="Survival datasets or not.")
parser.add_argument('--optuna', action='store_true', help="Hyperparameter tuning with optuna.")
parser.add_argument('--openml_id', type=int, help="Dataset ID for OpenML.")
parser.add_argument('--data_path', type=str, help="Path to the custom dataset CSV file.")
parser.add_argument('--dataset', type=str, choices=['metabric', 'support', 'flchain', 'gbsg'],
                    help="Choose a survival analysis dataset.")
parser.add_argument('--add_mask', action='store_true', help="Concatenate masks.")
parser.add_argument('--time_embedding_type', type=str, choices=['learnable','none'])
parser.add_argument('--dset_id', type=int)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', required=True, type=str,
                    choices=['binary', 'multiclass', 'regression', 'survival', 'deephit', 'time'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,
                    choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--dset_seed', default=5, type=int)
parser.add_argument('--active_log', action='store_true')
parser.add_argument('--decoder', action='store_true')

parser.add_argument('--use_focal', action='store_true')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                    choices=['contrastive', 'contrastive_sim', 'denoising'])
parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default=0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=2.0)

opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
elif opt.task == 'survival' or opt.task == 'deephit':
    opt.dtask = 'surv'
else:
    opt.dtask = 'clf'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb

    if opt.pretrain:
        wandb.init(project="saint_v2_all", group=opt.run_name,
                   name=f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task == 'multiclass':
            wandb.init(project="saint_v2_all_kamal", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group=opt.run_name,
                       name=f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')

print("Downloading and processing the dataset, it might take some time.")
if opt.use_openml:
    # Load dataset from OpenML
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(
        opt.openml_id, opt.set_seed, opt.task, datasplit=[0.65, 0.15, 0.2]
    )
elif opt.survival:
    # Load survival dataset
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, durations, X_train_full, y_train_full = preprocess_survival_dataset(
        opt.dataset, datasplit=[0.6, 0.2, 0.2], add_mask=opt.add_mask
    )
    NUM_DURATIONS = 3
    all_times = np.concatenate([
                    y_train["data"][:,0],
                    y_valid["data"][:,0],
                    y_test["data"][:,0]
                ])

    all_events =  np.concatenate([
                    y_train["data"][:,1],
                    y_valid["data"][:,1],
                    y_test["data"][:,1]
                ])           


    quantile_points = np.linspace(0, 1, NUM_DURATIONS + 2)[1:-1]
    time_grid = np.quantile(all_times[all_events == 1.0], quantile_points)

    if opt.task == 'deephit':
      import numpy as np

      from pycox.models import DeepHitSingle

      NUM_DURATIONS = 4
      labtrans = DeepHitSingle.label_transform(NUM_DURATIONS)

      # Fit only on training data
      labtrans.fit(y_train["data"][:,0], y_train["data"][:,1])

      # Compute index durations for each dataset
      idx_durations_train = labtrans.transform(y_train["data"][:,0], y_train["data"][:,1])[0]
      idx_durations_valid = labtrans.transform(y_valid["data"][:,0], y_valid["data"][:,1])[0]
      idx_durations_test  = labtrans.transform(y_test["data"][:,0], y_test["data"][:,1])[0]
      # Duration index (edges)
      duration_index = labtrans.cuts  # shape = NUM_DURATIONS+1
      NUM_DURATIONS = 3


continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_, nfeat = X_train_full['data'].shape
if nfeat > 100:
    opt.embedding_size = min(8, opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4, opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32, opt.embedding_size)
    opt.ff_dropout = 0.8

if opt.active_log:
    wandb.config.update(opt)

if opt.task == 'deephit':
    train_ds = DeepHitDataset(X_train, y_train, idx_durations_train, cat_idxs, continuous_mean_std)
    valid_ds = DeepHitDataset(X_valid, y_valid, idx_durations_valid, cat_idxs, continuous_mean_std)
    test_ds = DeepHitDataset(X_test, y_test, idx_durations_test, cat_idxs, continuous_mean_std)
else:
    train_full_ds = DataSetCatCon(X_train_full, y_train_full, cat_idxs, opt.dtask, continuous_mean_std)
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)

trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

if opt.task == 'regression' or opt.task == 'survival' :
    y_dim = 1
elif opt.task == "deephit" or opt.task == "time":
    y_dim = NUM_DURATIONS
else:
    y_dim = len(np.unique(y_train['data'][:, 1]))


cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(
    int)  # Appending 1 for CLS token, this is later used to generate embeddings.

if opt.task == 'deephit' or opt.task == 'time':
  dim_out = NUM_DURATIONS
else:
  dim_out = 1
print("Model output dim_out:", dim_out)

def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Optuna Trial] Using device: {device}")

    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    embedding_size = trial.suggest_categorical("embedding_size", [8, 16, 32])
    batch_size = trial.suggest_categorical("batchsize", [32, 64, 128, 256])
    transformer_depth = trial.suggest_int("transformer_depth", 1, 6)
    attention_heads = trial.suggest_int("attention_heads", 2, 8)
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.5)
    ff_dropout = trial.suggest_float("ff_dropout", 0.1, 0.5)
    optimizer_type = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "linear"])
    epochs = trial.suggest_int("epochs", 20, 100)
    # Optional DeepHit loss hyperparameters
    if opt.task == "deephit":
        alpha = trial.suggest_float("alpha", 0.2, 0.8)
        sigma = trial.suggest_float("sigma", 0.01, 0.5)
        dim_out = NUM_DURATIONS
        y_dim = NUM_DURATIONS
    elif opt.task == "time":
        alpha = trial.suggest_categorical("alpha", [0.5, 1.0, 2.0])
        beta = trial.suggest_categorical("beta", [0.05, 0.1, 0.2])
        gamma = trial.suggest_categorical("gamma", [1.0, 2.0, 3.0])
        use_focal = trial.suggest_categorical("use_focal", [True, False])
        dim_out = NUM_DURATIONS
        y_dim = NUM_DURATIONS
    else:
        dim_out = 1
        y_dim = 1

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=embedding_size,
        dim_out=dim_out,
        depth=transformer_depth,
        heads=attention_heads,
        attn_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=y_dim
    ).to(device)

    if opt.pretrain:
        from pretraining import SAINT_pretrain
        model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device)

    # Optimizer
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if opt.task == "deephit":
        criterion = DeepHitSingleLoss(alpha=alpha, sigma=sigma).to(device)
    elif opt.task == "time":
        criterion = CombinedLossMonotonic(alpha=alpha, beta=beta)
    else:
        criterion = CoxPHLoss().to(device)

    # Training loop
    best_valid_ibs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for batch in trainloader:
            optimizer.zero_grad()

            if opt.task == "deephit":
                x_categ, x_cont, durations, idx_durations, events, rank_mat, cat_mask, con_mask = [d.to(device) for d in batch]
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                reps = model.transformer(x_categ_enc, x_cont_enc)
                phi = model.mlpfory(reps[:, 0, :])
                idx_durations = idx_durations.view(-1, 1).long()
                rank_mat = pair_rank_mat(durations, events)
                loss = criterion(phi, idx_durations, events, rank_mat)
            elif opt.task == 'time':
                x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in batch]
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                eval_times = torch.tensor(time_grid, dtype=torch.float32)
                time_model = SAINTWithTime(
                        base_saint_model=model,
                        dim=embedding_size,
                        time_points=time_grid,
                        time_embedding_type=opt.time_embedding_type    # or 'learnable' or 'sin'
                    ).to(device)
                batch_size = x_categ.shape[0]
                time_ids = eval_times[None, :].repeat(batch_size, 1)  # (B, T)
                hazards = time_model.forward(x_categ_enc, x_cont_enc, time_ids)
                loss, survival, hazard_increments  = criterion(hazards, y_gts[:, 0], y_gts[:, 1], time_grid)
            else:
                x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in batch]
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                              
                reps = model.transformer(x_categ_enc, x_cont_enc)

                y_outs = model.mlpfory(reps[:, 0, :])
                loss = criterion(y_outs, y_gts)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            if opt.task == "deephit":
                predictor = SAINTDeepHitPredictor(model, duration_index, device, NUM_DURATIONS)
                val_metrics = predictor.evaluate(validloader, y_valid["data"][:, 0], y_valid["data"][:, 1], time_grid)
                val_ibs = val_metrics["Integrated Brier Score"]
            elif opt.task == "time":
                from utils import metrics_surv_saintwithtime
                # Skip epoch if any metric is NaN
                valid_metrics = metrics_surv_saintwithtime( time_model, validloader, device, time_grid, vision_dset=False)
                val_ibs = valid_metrics["Integrated Brier Score"]
            else:
                val_ibs = ibs(opt.dataset, model, validloader, device, False, time_grid)

            if val_ibs is not None and not np.isnan(val_ibs) and val_ibs < best_valid_ibs:
                best_valid_ibs = val_ibs
                torch.save(model.state_dict(), "best_model_optuna.pth")

            trial.report(val_ibs, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_valid_ibs  # minimize ibs

# Ensure all tensors are moved to the correct device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if opt.optuna:
  # Run Optuna optimization for survival analysis
  sampler = optuna.samplers.TPESampler(seed=10)
  study = optuna.create_study(direction="minimize", sampler=sampler)  # Maximize concordance index
  def safe_objective(trial):
            try:
                return objective(trial)
            except (ValueError, RuntimeError) as e:
                print(f"[Trial Pruned] Error: {e}")
                raise optuna.exceptions.TrialPruned()

  study.optimize(safe_objective, n_trials=100)

  # Save best hyperparameters
  best_params = study.best_params
  print("Best hyperparameters for survival analysis:", best_params)

  # Save best params to a file
  import json

  with open("best_hyperparams_survival.json", "w") as f:
      json.dump(best_params, f, indent=4)

# Train model 10 times with different seeds
brier_scores = []
all_test_metrics = []
for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if opt.optuna:
      if opt.dataset == "custom":
        embedding_size = opt.embedding_size
      else:
        embedding_size = best_params['embedding_size']
        if opt.task =="deephit":
          y_dim = NUM_DURATIONS
        else:
          y_dim = 1
      model = SAINT(
          categories=tuple(cat_dims),
          num_continuous=len(con_idxs),
          dim=embedding_size,
          dim_out=dim_out,
          depth=best_params['transformer_depth'],
          heads=best_params['attention_heads'],
          attn_dropout=best_params['attention_dropout'],
          ff_dropout=best_params['ff_dropout'],
          mlp_hidden_mults=(4, 2),
          cont_embeddings=opt.cont_embeddings,
          attentiontype=opt.attentiontype,
          final_mlp_style=opt.final_mlp_style,
          y_dim=y_dim
      ).to(device)

    else:
      embedding_size = opt.embedding_size
      if opt.task == "deephit":
        model = SAINT(
          categories=tuple(cat_dims),
          num_continuous=len(con_idxs),
          dim=opt.embedding_size,
          dim_out=dim_out,
          depth=opt.transformer_depth,
          heads=opt.attention_heads,
          attn_dropout=opt.attention_dropout,
          ff_dropout=opt.ff_dropout,
          mlp_hidden_mults=(4, 2),
          cont_embeddings=opt.cont_embeddings,
          attentiontype=opt.attentiontype,
          final_mlp_style=opt.final_mlp_style,
          y_dim=NUM_DURATIONS
      ).to(device)
      else:
        model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=len(con_idxs),
            dim=opt.embedding_size,
            dim_out=dim_out,
            depth=opt.transformer_depth,
            heads=opt.attention_heads,
            attn_dropout=opt.attention_dropout,
            ff_dropout=opt.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=opt.cont_embeddings,
            attentiontype=opt.attentiontype,
            final_mlp_style=opt.final_mlp_style,
            y_dim=1
        ).to(device)
    vision_dset = opt.vision_dset

    if y_dim == 2 and opt.task == 'binary':
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train['data'][:, 0]), y=y_train['data'][:, 0])
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    elif y_dim > 2 and opt.task == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(device)
    elif opt.task == 'regression':
        criterion = nn.MSELoss().to(device)
    elif opt.task == 'survival':
        criterion = CoxPHLoss().to(device)
    elif opt.task == 'time':
      if opt.optuna:
        alpha = best_params["alpha"]
        beta = best_params["beta"]
      else:
        alpha = opt.alpha
        beta = opt.beta
      criterion = CombinedLossMonotonic(alpha=alpha, beta=beta)
    elif opt.task == 'deephit':
      if opt.optuna:
        alpha = best_params["alpha"]
        sigma = best_params["sigma"]
      else:
        alpha = opt.alpha
        sigma = opt.sigma
      criterion = DeepHitSingleLoss(alpha=alpha, sigma=sigma).to(device)
    else:
        raise 'case not written yet'

    model.to(device)

    if opt.pretrain:
        from pretraining import SAINT_pretrain
        model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device)

    ## Choosing the optimizer

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                              momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler

        scheduler = get_scheduler(opt, optimizer)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    best_valid_auroc = 0
    best_valid_accuracy = 0
    best_test_auroc = 0
    best_test_accuracy = 0
    best_valid_rmse = 100000
    best_valid_ibs = 100
    best_test_ibs = 100
    best_test_metrics = []
    if opt.optuna:
      epochs = best_params['epochs']
    else:
      epochs = opt.epochs
    print('Training begins now.')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            if opt.task == 'deephit':
              # Unpack the batch
              x_categ, x_cont, durations, idx_durations, events, rank_mat, cat_mask, con_mask = [d.to(device) for d in data]

              # Embed inputs
              _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
              reps = model.transformer(x_categ_enc, x_cont_enc)
              phi = model.mlpfory(reps[:, 0, :])

              # Ensure duration indices are shaped correctly
              idx_durations = idx_durations.view(-1, 1).long()
              rank_mat = pair_rank_mat(durations, events)

              # Compute DeepHit loss
              loss = criterion(phi, idx_durations, events, rank_mat)
              loss.backward()
              optimizer.step()
              if opt.optimizer == 'SGD':
                  scheduler.step()
              running_loss += loss.item()
            elif opt.task == 'time':
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
                  3].to(device), data[4].to(device)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                eval_times = torch.tensor(time_grid, dtype=torch.float32)
                time_model = SAINTWithTime(
                        base_saint_model=model,
                        dim=embedding_size,
                        time_points=time_grid,
                        time_embedding_type=opt.time_embedding_type    # or 'learnable' or 'sin'
                    ).to(device)
                # During forward pass:
                batch_size = x_categ.shape[0]
                time_ids = eval_times[None, :].repeat(batch_size, 1)  # (B, T)
                hazards= time_model.forward(x_categ_enc, x_cont_enc, time_ids)
                loss, survival, hazard_increments = criterion(hazards, y_gts[:, 0], y_gts[:, 1], time_grid)
                loss.backward()
                optimizer.step()
                if opt.optimizer == 'SGD':
                    scheduler.step()
                running_loss += loss.item()
            else:
              # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont.
              x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
                  3].to(device), data[4].to(device)
              # We are converting the data to embeddings in the next step
              _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
              reps = model.transformer(x_categ_enc, x_cont_enc)
              # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
              y_reps = reps[:, 0, :]
              y_outs = model.mlpfory(y_reps)
              if opt.task == 'regression' or opt.task == 'survival' or opt.task == 'survival_survloss':
                  loss = criterion(y_outs, y_gts)
              else:
                  loss = criterion(y_outs, y_gts.squeeze().long())
              loss.backward()
              optimizer.step()
              if opt.optimizer == 'SGD':
                  scheduler.step()
              running_loss += loss.item()

        if opt.active_log:
            wandb.log({'epoch': epoch, 'train_epoch_loss': running_loss,
                       'loss': loss.item()
                       })
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                if opt.task == 'survival':
                    from utils import metrics_surv
                    decoder_Model=None
                    valid_ibs = ibs(opt.dataset, model, validloader, device, vision_dset, time_grid, decoder =opt.decoder, decoder_model=decoder_Model)
                    test_ibs = ibs(opt.dataset, model, testloader, device, vision_dset, time_grid, decoder =opt.decoder, decoder_model=decoder_Model)
                     # Skip epoch if any metric is NaN
                    if np.isnan(valid_ibs) or np.isnan(test_ibs):
                        print(f"[EPOCH {epoch + 1}] Skipped due to NaN in c-index.")
                        continue
                    valid_metrics = metrics_surv(opt.dataset, model, validloader, device, vision_dset, time_grid, decoder =opt.decoder, decoder_model=decoder_Model)
                    test_metrics = metrics_surv(opt.dataset, model, testloader, device, vision_dset, time_grid, decoder =opt.decoder, decoder_model=decoder_Model)

                    print(f"[EPOCH {epoch + 1}] VALID METRICS:")
                    print(valid_ibs)
                    print(f"[EPOCH {epoch + 1}] TEST METRICS:")
                    print(test_ibs)
                  

                    if opt.active_log:
                        wandb.log({f"valid_{k}": v for k, v in valid_metrics.items()})
                        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

                    if valid_ibs < best_valid_ibs:
                        best_valid_ibs = valid_ibs
                        best_test_ibs = test_ibs
                        best_test_metrics = test_metrics
                        torch.save(model.state_dict(), f"{modelsave_path}/bestmodel.pth")
    
                elif opt.task == 'survival_survloss':
                    valid_mae = mean_abs_error(model, validloader, device, vision_dset)
                    test_mae = mean_abs_error(model, testloader, device, vision_dset)
                    print('[EPOCH %d] VALID MAE: %.3f' %
                          (epoch + 1, valid_mae))
                    print('[EPOCH %d] TEST MAE: %.3f' %
                          (epoch + 1, test_mae))
                    if opt.active_log:
                        wandb.log({'valid_mae': valid_mae, 'test_mae': test_mae})
                    if valid_mae < best_valid_mae:
                        best_valid_mae = valid_mae
                        best_test_mae = test_mae
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
                elif opt.task == 'time':
                  from utils import metrics_surv_saintwithtime
                  # Skip epoch if any metric is NaN
                  valid_metrics = metrics_surv_saintwithtime( time_model, validloader, device,  time_grid, vision_dset)
                  test_metrics = metrics_surv_saintwithtime( time_model, testloader, device,  time_grid, vision_dset)
                  valid_ibs = valid_metrics["Integrated Brier Score"]
                  test_ibs = test_metrics["Integrated Brier Score"]
                  if np.isnan(valid_ibs) or np.isnan(test_ibs):
                      print(f"[EPOCH {epoch + 1}] Skipped due to NaN in c-index.")
                      continue
                  print(f"[EPOCH {epoch + 1}] VALID METRICS:")
                  print(valid_ibs)
                  print(f"[EPOCH {epoch + 1}] TEST METRICS:")
                  print(test_ibs)
                
                  if opt.active_log:
                      wandb.log({f"valid_{k}": v for k, v in valid_metrics.items()})
                      wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

                  if valid_ibs < best_valid_ibs:
                      best_valid_ibs = valid_ibs
                      best_test_ibs = test_ibs
                      best_test_metrics = test_metrics
                      torch.save(model.state_dict(), f"{modelsave_path}/bestmodel.pth")

                elif opt.task == 'deephit':

                  predictor = SAINTDeepHitPredictor(model, duration_index, device, NUM_DURATIONS)

                  durations_val = y_valid['data'][:, 0]
                  events_val = y_valid['data'][:, 1]
                  durations_test = y_test['data'][:, 0]
                  events_test = y_test['data'][:, 1]

                  valid_metrics = predictor.evaluate(validloader, durations_val, events_val, time_grid, log_to_wandb=opt.active_log)
                  test_metrics = predictor.evaluate(testloader, durations_test, events_test, time_grid, log_to_wandb=opt.active_log)

                  print(f"[EPOCH {epoch + 1}] VALID METRICS:")
                  print(valid_metrics)
                  print(f"[EPOCH {epoch + 1}] TEST METRICS:")
                  print(test_metrics)

                  valid_ibs = valid_metrics["Integrated Brier Score"]  # You may switch to test_metrics["C-Index"] if preferred
                  test_ibs = test_metrics["Integrated Brier Score"]
                  if np.isnan(valid_ibs):
                      print(f"[EPOCH {epoch + 1}] Skipped due to NaN in c-index.")
                      continue

                  if valid_ibs < best_valid_ibs:
                      best_valid_ibs = valid_metrics["Integrated Brier Score"]
                      best_test_ibs = test_ibs
                      best_test_metrics = test_metrics
                      torch.save(model.state_dict(), f"{modelsave_path}/bestmodel.pth")

            model.train()
    
    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
    if opt.task in ['survival', 'deephit', 'time']:
        print('IBS on best model:  %.3f' % (best_test_ibs))
        if best_test_ibs != 100:
          brier_scores.append(best_test_ibs)
          all_test_metrics.append(best_test_metrics)
        print(f"Run {seed + 1}: Best IBS = {best_test_ibs:.3f}")
    if opt.active_log:
        if opt.task in ['survival', 'deephit','time']:
            wandb.log({'total_parameters': total_parameters, 'test_ibs_bestep': best_test_ibs,
                       'cat_dims': len(cat_idxs), 'con_dims': len(con_idxs)})

# Calculate mean and standard deviation of C-Index
import collections

if opt.task in ['survival', 'deephit', 'time']:
    metrics_keys = all_test_metrics[0].keys()
    aggregated = collections.defaultdict(list)

    for run_metrics in all_test_metrics:
        for k in metrics_keys:
            aggregated[k].append(run_metrics[k])

    for k in metrics_keys:
        values = aggregated[k]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{k} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        if opt.active_log:
            wandb.log({f"mean_{k}": mean_val, f"std_{k}": std_val})



