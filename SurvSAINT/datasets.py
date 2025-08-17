from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pycox.datasets import metabric, support, flchain, gbsg
from sksurv.datasets import load_flchain
from torch.utils.data import Dataset

def get_idx_durations(durations, duration_index):
    bins = np.digitize(durations, duration_index[1:-1])
    return np.clip(bins, 0, len(duration_index) - 2)  # max index allowed

class DeepHitDataset(Dataset):
    def __init__(self, X, Y, idx_durations, cat_cols, continuous_mean_std=None):
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X_data = X['data'].copy()
        con_cols = list(set(np.arange(X_data.shape[1])) - set(cat_cols))

        self.X1 = X_data[:, cat_cols].astype(np.int64)
        self.X2 = X_data[:, con_cols].astype(np.float32)
        self.X1_mask = X_mask[:, cat_cols].astype(np.int64)
        self.X2_mask = X_mask[:, con_cols].astype(np.int64)
        self.durations = Y['data'][:, 0].astype(np.float32)
        self.events = Y['data'][:, 1].astype(np.float32)
        self.idx_durations = idx_durations.astype(np.int64)
        self.cls = np.zeros((self.durations.shape[0], 1), dtype=int)
        self.cls_mask = np.ones((self.durations.shape[0], 1), dtype=int)
        self.rank_mat = self.compute_rank_matrix(self.durations, self.events)

    def compute_rank_matrix(self, durations, events):
        n = len(durations)
        rank_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if durations[i] < durations[j] and events[i] == 1:
                    rank_mat[i, j] = 1.0
        return rank_mat

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, idx):
        x_categ = np.concatenate((self.cls[idx], self.X1[idx]))
        cat_mask = np.concatenate((self.cls_mask[idx], self.X1_mask[idx]))
        return (
            x_categ,
            self.X2[idx],
            self.durations[idx],
            self.idx_durations[idx],
            self.events[idx],
            self.rank_mat[idx],
            cat_mask,
            self.X2_mask[idx]
        )

def preprocess_survival_dataset(dataset_name, datasplit=[0.6, 0.2, 0.2], add_mask=False):
    if dataset_name == 'metabric':
        df = metabric.read_df()
        categorical_cols = ['x4', 'x5', 'x6', 'x7']
        continuous_cols = [col for col in df.columns if col not in categorical_cols + ['duration', 'event']]
    elif dataset_name == 'support':
        df = support.read_df()
        categorical_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        continuous_cols = [col for col in df.columns if col not in categorical_cols + ['duration', 'event']]
    elif dataset_name == 'flchain':
        df, target = load_flchain()
        df = df.drop(columns=['chapter'])
        # Extract durations and events from y
        durations = [entry[1] for entry in target]
        events = [entry[0] for entry in target]
        durations = np.array(durations)
        events = np.array(events)
        categorical_cols = ['sex', 'sample.yr', 'flc.grp', 'mgus']
        continuous_cols = [col for col in df.columns if col not in categorical_cols]
    elif dataset_name == 'gbsg':
        df = gbsg.read_df()
        categorical_cols = ['x0', 'x1', 'x2', 'x4']
        continuous_cols = [col for col in df.columns if col not in categorical_cols + ['duration', 'event']]
    else:
        raise ValueError("Invalid dataset choice.")


    if dataset_name == 'flchain':
      X = df
      y = pd.DataFrame({
          'duration': durations,
          'event': events
      })
    else:
    # Extract features and targets
      X = df.drop(columns=['duration', 'event'])
      y = df[['duration', 'event']]
    durations = y['duration'].values
    nan_mask = (~X.isna()).astype(int)

    for col in categorical_cols:
      if pd.api.types.is_categorical_dtype(X[col]):
          if "MissingValue" not in X[col].cat.categories:
              X[col] = X[col].cat.add_categories(["MissingValue"])
          X[col] = X[col].fillna("MissingValue")
          X[col] = X[col].astype(str)
      else:
          X[col] = X[col].fillna("MissingValue").astype(str)

      le = LabelEncoder()
      X[col] = le.fit_transform(X[col].values)

    # === Split indices (NOT the data yet) for train_val/test
    trainval_idx, test_idx, y_train_full, y_test= train_test_split(
        X.index, y, test_size=0.2, random_state=42, stratify=y['event']
    )

    train_idx, val_idx, y_train, y_val = train_test_split(
        trainval_idx, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full['event']
    )

    # Impute using training set mean only
    train_mean = X.loc[train_idx, continuous_cols].mean()
    X.loc[train_idx, continuous_cols] = X.loc[train_idx, continuous_cols].fillna(train_mean)
    X.loc[val_idx, continuous_cols] = X.loc[val_idx, continuous_cols].fillna(train_mean)
    X.loc[test_idx, continuous_cols] = X.loc[test_idx, continuous_cols].fillna(train_mean)

    # Step 1: Track the original column names (before concatenation)
    original_feature_cols = X.columns.tolist()  # Assumes X is still a DataFrame here

    scaler = StandardScaler()
    train_val_mean = X.loc[train_idx, continuous_cols].mean()
    train_val_std = X.loc[train_idx, continuous_cols].std().replace(0, 1e-6)
    X.loc[train_idx, continuous_cols] = scaler.fit_transform(X.loc[train_idx, continuous_cols])
    X.loc[test_idx, continuous_cols] = scaler.transform(X.loc[test_idx, continuous_cols])
    X.loc[val_idx, continuous_cols] = scaler.transform(X.loc[val_idx, continuous_cols])

    # Add the mask
    if add_mask:
        mask_cols = [f"mask_{col}" for col in original_feature_cols]
        mask_df = pd.DataFrame(nan_mask.values.astype(np.float32), columns=mask_cols, index=X.index)
        X = pd.concat([X, mask_df], axis=1)
        # Update categorical column list
        total_continuous = continuous_cols + mask_cols
        nan_mask = (~X.isna()).astype(int)

    cat_idxs = [X.columns.get_loc(col) for col in categorical_cols]

    if not add_mask:
      total_continuous = continuous_cols
    cont_idxs = [X.columns.get_loc(col) for col in total_continuous]

    X_train_full = X.loc[trainval_idx]
    X_test = X.loc[test_idx]
    y_train_full = y.loc[trainval_idx]
    y_test = y.loc[test_idx]
    mask_train_full = nan_mask.loc[trainval_idx]
    mask_test = nan_mask.loc[test_idx]

    X_train = X.loc[train_idx]
    X_valid = X.loc[val_idx]
    y_train = y.loc[train_idx]
    y_valid = y.loc[val_idx]
    mask_train = nan_mask.loc[train_idx]
    mask_valid = nan_mask.loc[val_idx]

    print("proper mask")
    # Convert to dictionary format
    def to_dict(X, y, mask):
        nan_mask = (~pd.isna(X)).astype(int)
        return {
            'data': X.values,
            'mask': mask.values
        }, {'data': y.values}

    X_train_full_dict, y_train_full_dict = to_dict(X_train_full, y_train_full, mask_train_full)
    X_train_dict, y_train_dict = to_dict(X_train, y_train, mask_train)
    X_valid_dict, y_valid_dict = to_dict(X_valid, y_valid, mask_valid)
    X_test_dict, y_test_dict = to_dict(X_test, y_test, mask_test)

    # Calculate categorical dimensions
    cat_dims = [len(X[col].unique()) for col in categorical_cols]

    return cat_dims, cat_idxs, cont_idxs, X_train_dict, y_train_dict, X_valid_dict, y_valid_dict, X_test_dict, y_test_dict, train_val_mean, train_val_std, durations, X_train_full_dict, y_train_full_dict
