from pycox.datasets import metabric, nwtco, support, gbsg, flchain
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sksurv.datasets import load_flchain
from .utils import LabelTransform
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(config):
    data = config['data']
    horizons = config['horizons']
    assert data in ["metabric", "nwtco", "support", "gbsg", "flchain", "seer", "custom", "custom_agg"], "Data Not Found!"
    get_target = lambda df: (df['duration'].values, df['event'].values)

    if data == "metabric":
        df = metabric.read_df()
        cols_categorical = ["x4", "x5", "x6", "x7"]
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']

    elif data == "support":
        df = support.read_df()
        cols_categorical = ["x1", "x2", "x3", "x4", "x5", "x6"]
        cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

    elif data == "gbsg":
        df = gbsg.read_df()
        cols_categorical = ['x0', 'x1', 'x2', 'x4']
        cols_standardize = [c for c in df.columns if c not in cols_categorical + ['duration', 'event']]

    elif data == "flchain":
        df_raw, target = load_flchain()
        df = df_raw.copy()
        df['duration'] = [x[1] for x in target]
        df['event'] = [x[0] for x in target]
        df = df.drop(columns=['chapter'])
        cols_categorical = ['sex', 'sample.yr', 'flc.grp', 'mgus']
        cols_standardize = [c for c in df.columns if c not in cols_categorical + ['duration', 'event']]

    else:
        raise ValueError(f"Unsupported dataset: {data}")

    # Preprocessing
    df_feat = df.drop(columns=['duration', 'event'])
    mask_num = ~df_feat[cols_standardize].isna()
    mask_cat = ~df_feat[cols_categorical].isna()
    missing_mask = pd.concat([mask_num, mask_cat], axis=1).astype(float)

    missing_mask.shape[1] == df_feat[cols_categorical].shape[1] + df_feat[cols_standardize].shape[1]


    for col in cols_categorical:
      if pd.api.types.is_categorical_dtype(df_feat[col]):
          if "MissingValue" not in df_feat[col].cat.categories:
              df_feat[col] = df_feat[col].cat.add_categories(["MissingValue"])
          df_feat[col] = df_feat[col].fillna("MissingValue")
          df_feat[col] = df_feat[col].astype(str)
      else:
          df_feat[col] = df_feat[col].fillna("MissingValue").astype(str)


    df_target_full = df[['duration', 'event']]

    # --- Impute numerical columns using mean from trainval (X_temp)
    X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
        df.index, df_target_full, test_size=0.20, random_state=42, stratify=df_target_full["event"]
    )
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_temp_idx, y_temp.loc[X_temp_idx], test_size=0.20, random_state=42, stratify=y_temp.loc[X_temp_idx]["event"]
    )

    # === Standardize numeric cols (only using trainval mean/std)
    scaler = StandardScaler()
    df_feat_standardize = df_feat[cols_standardize]
    train_mean = df_feat_standardize.loc[X_train_idx].mean()

    df_feat_standardize.loc[X_train_idx] = df_feat_standardize.loc[X_train_idx].fillna(train_mean)
    df_feat_standardize.loc[X_val_idx] = df_feat_standardize.loc[X_val_idx].fillna(train_mean)
    df_feat_standardize.loc[X_test_idx] = df_feat_standardize.loc[X_test_idx].fillna(train_mean)

    df_feat_standardize.loc[X_train_idx] = scaler.fit_transform(df_feat_standardize.loc[X_train_idx])
    df_feat_standardize.loc[X_val_idx] = scaler.transform(df_feat_standardize.loc[X_val_idx])
    df_feat_standardize.loc[X_test_idx] = scaler.transform(df_feat_standardize.loc[X_test_idx])
    
    df_feat_standardize_disc = df_feat_standardize.astype(float)

    if config.add_mask:
      df_feat_standardize_disc = pd.concat([df_feat_standardize_disc, missing_mask], axis=1)
      mask_columns = missing_mask.columns.tolist()
      cols_standardize = cols_standardize + mask_columns
      
    vocab_size = 0
    for feat in cols_categorical:
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1

    df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
    
    if config.add_mask:
      nan_mask_df = (~missing_mask.isna()).astype(int)
      missing_mask = pd.concat([missing_mask, nan_mask_df], axis=1).astype(float)

    # 64/16/20 split
    df_feat_full = df_feat

    X_temp = df_feat_full.loc[X_temp_idx]
    X_train = df_feat_full.loc[X_train_idx]
    X_val   = df_feat_full.loc[X_val_idx]
    X_test  = df_feat_full.loc[X_test_idx]
    
    y_temp = df_target_full.loc[X_temp_idx]
    y_train = df_target_full.loc[X_train_idx]
    y_val   = df_target_full.loc[X_val_idx]
    y_test  = df_target_full.loc[X_test_idx]

    # Get cutoffs and label transform
    df_all = df.copy()
    times = np.quantile(df_all["duration"][df_all["event"] == 1.0], horizons).tolist()
    cuts = np.array([df_all["duration"].min()] + times + [df_all["duration"].max()])
    labtrans = LabelTransform(cuts=cuts)
    labtrans.fit(*get_target(df_all.loc[X_train.index]))

    # Transform full dataset with label transform
    y_all = labtrans.transform(*get_target(df_all))

    y_temp_trans = (
        y_all[0][X_temp.index],
        y_all[1][X_temp.index],
        y_all[2][X_temp.index]
    )


    df_y_trainval_trans = pd.DataFrame({
                "duration": y_all[0][X_temp.index],
                "event": y_all[1][X_temp.index],
                "proportion": y_all[2][X_temp.index]
            }, index=X_temp.index)
    
    # Transformed labels
    df_y_train_trans = pd.DataFrame({
        "duration": y_all[0][X_train.index],
        "event": y_all[1][X_train.index],
        "proportion": y_all[2][X_train.index]
    }, index=X_train.index)

    df_y_val_trans = pd.DataFrame({
        "duration": y_all[0][X_val.index],
        "event": y_all[1][X_val.index],
        "proportion": y_all[2][X_val.index]
    }, index=X_val.index)

    # Raw labels (for evaluation)
    df_y_trainval_raw = df_all.loc[X_temp.index, ['duration', 'event']]
    df_trainval = df_all.loc[X_temp.index]
    df_y_train_raw = df_all.loc[X_train.index, ['duration', 'event']]
    df_y_val_raw = df_all.loc[X_val.index, ['duration', 'event']]
    df_y_test_raw = df_all.loc[X_test.index, ['duration', 'event']]

    config['cols_standardize'] = cols_standardize
    config['labtrans'] = labtrans
    config['num_numerical_feature'] = int(len(cols_standardize))
    config['num_categorical_feature'] = int(len(cols_categorical))
    config['num_feature'] = int(len(X_train.columns))
    config['vocab_size'] = int(vocab_size)
    config['duration_index'] = labtrans.cuts
    config['out_feature'] = int(labtrans.out_features)
    config['num_event'] = 1
    print(labtrans.cuts)  

    return (
        df_all,
        X_train, df_y_train_trans, df_y_train_raw,
        X_test, df_y_test_raw,
        X_val, df_y_val_trans, df_y_val_raw,
        missing_mask, X_temp, df_y_trainval_trans, df_y_trainval_raw, y_temp_trans, df_trainval
    )
