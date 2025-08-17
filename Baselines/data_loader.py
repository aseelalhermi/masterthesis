from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
import numpy as np
from pycox.datasets import metabric, support, flchain, gbsg
from sksurv.datasets import load_flchain

def load_dataset(dataset_name, add_mask=True):
    # === Load dataset
    if dataset_name == "metabric":
        df = metabric.read_df()
    elif dataset_name == "support":
        df = support.read_df()
    elif dataset_name == 'flchain':
        df, target = load_flchain()
        df = df.drop(columns=['chapter']) 
    elif dataset_name == 'gbsg':
        df = gbsg.read_df()
    else:
        raise ValueError("Invalid dataset name provided.")

    # === Parse durations/events

    if dataset_name == "flchain":
        durations = np.array([entry[1] for entry in target])
        events = np.array([entry[0] for entry in target])
    else:
        durations = df['duration'].values
        events = df['event'].values
        df = df.drop(columns=['duration', 'event'])

    # === Define categorical columns
    if dataset_name == "metabric":
        cat_cols = ['x4', 'x5', 'x6', 'x7']
    elif dataset_name == "support":
        cat_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    elif dataset_name == "gbsg":
        cat_cols = ['x0', 'x1', 'x2', 'x4']
    elif dataset_name == "flchain":
        cat_cols = ['sex', 'sample.yr', 'flc.grp', 'mgus']

    num_cols = [col for col in df.columns if col not in cat_cols]
    original_cols = num_cols + cat_cols
    # === Build mask matrix
    mask_df = df[original_cols].isna().astype(float).rsub(1.0)

    # === Impute missing values
    df[cat_cols] = SimpleImputer(strategy="constant", fill_value="MissingValue").fit_transform(df[cat_cols].astype(str))
    
    # === Split indices (NOT the data yet) for train_val/test
    trainval_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=events
    )

    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, random_state=42, stratify=events[trainval_idx]
    )

    # === Impute numeric columns: fit on train only
    num_imputer = SimpleImputer(strategy="mean")
    df.loc[train_idx, num_cols] = num_imputer.fit_transform(df.loc[train_idx, num_cols])
    df.loc[val_idx, num_cols] = num_imputer.transform(df.loc[val_idx, num_cols])
    df.loc[test_idx, num_cols] = num_imputer.transform(df.loc[test_idx, num_cols])

    # === Fit StandardScaler only on train_val numeric features
    scaler = StandardScaler()
    df.loc[train_idx, num_cols] = scaler.fit_transform(df.loc[train_idx, num_cols])
    df.loc[val_idx, num_cols] = scaler.transform(df.loc[val_idx, num_cols])
    df.loc[test_idx, num_cols] = scaler.transform(df.loc[test_idx, num_cols])

    # === Encode features
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    x = ct.fit_transform(df)

    # === Build mask

    if add_mask:
        if hasattr(x, "toarray"):  # in case it's a sparse matrix
            x = x.toarray()
        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)  # ensure 2D for hstack
        x = np.hstack([x, mask_df.values])
    

    # === Now split actual data using previously saved indices
    x_train_val = x[trainval_idx]
    x_test = x[test_idx]
    durations_train_val = durations[trainval_idx]
    durations_test = durations[test_idx]
    events_train_val = events[trainval_idx]
    events_test = events[test_idx]

    x_train = x[train_idx]
    x_val = x[val_idx]
    durations_train = durations[train_idx]
    durations_val = durations[val_idx]
    events_train = events[train_idx]
    events_val = events[val_idx]


    survival_train = Surv.from_arrays(events_train, durations_train)
    survival_val = Surv.from_arrays(events_val, durations_val)
    survival_test = Surv.from_arrays(events_test, durations_test)
    survival_trainval = Surv.from_arrays(events_train_val, durations_train_val)

    return x_train, x_val, x_test, survival_train, durations_val, events_val, survival_test, x_train_val, durations_train_val, events_train_val

def load_dataset_tokenized(dataset_name, custom_data_path=None):

    if dataset_name == "metabric":
        df = metabric.read_df()
    elif dataset_name == "support":
        df = support.read_df()
    elif dataset_name == 'flchain':
        df, target = load_flchain()
        df = df.drop(columns=['chapter'])
    elif dataset_name == 'gbsg':
        df = gbsg.read_df()
    else:
        raise ValueError("Invalid dataset name provided.")


    if dataset_name == "flchain":
        durations = np.array([entry[1] for entry in target])
        events = np.array([entry[0] for entry in target])
    else:
        durations = df['duration'].values
        events = df['event'].values
        df = df.drop(columns=['duration', 'event'])

    if dataset_name == "metabric":
        cat_cols = ['x4', 'x5', 'x6', 'x7']
    elif dataset_name == "support":
        cat_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    elif dataset_name == "gbsg":
        cat_cols = ['x0', 'x1', 'x2', 'x4']
    elif dataset_name == "flchain":
        cat_cols = ['sex', 'sample.yr', 'flc.grp', 'mgus']

    num_cols = [col for col in df.columns if col not in cat_cols]
    original_cols = num_cols + cat_cols

    # === Build mask matrix
    mask_df = df[original_cols].isna().astype(float).rsub(1.0)
    # Impute
    df[cat_cols] = SimpleImputer(strategy="constant", fill_value="MissingValue").fit_transform(df[cat_cols].astype(str))


    # === Split indices (NOT the data yet) for train_val/test
    trainval_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=events
    )

    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, random_state=42, stratify=events[trainval_idx]
    )

    # === Impute numeric columns: fit on train only
    num_imputer = SimpleImputer(strategy="mean")
    df.loc[train_idx, num_cols] = num_imputer.fit_transform(df.loc[train_idx, num_cols])
    df.loc[val_idx, num_cols] = num_imputer.transform(df.loc[val_idx, num_cols])
    df.loc[test_idx, num_cols] = num_imputer.transform(df.loc[test_idx, num_cols])

    # Scale numeric
    scaler = StandardScaler()
    df.loc[train_idx, num_cols] = scaler.fit_transform(df.loc[train_idx, num_cols])
    df.loc[test_idx, num_cols] = scaler.transform(df.loc[test_idx, num_cols])
    df.loc[val_idx, num_cols] = scaler.transform(df.loc[val_idx, num_cols])
    x_num = df[num_cols].values.astype(np.float32)
    x = np.hstack([x_num, mask_df.values.astype(np.float32)])
    # Encode categories with tokens
    x_cat_tokens = []
    cat_cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        x_cat_tokens.append(df[col].values)
        cat_cardinalities.append(len(le.classes_))

    x_cat_tokens = np.stack(x_cat_tokens, axis=1).astype(np.int64)

    x_train_val_num = x[trainval_idx]
    x_test_num = x[test_idx]
    x_train_val_cat = x_cat_tokens[trainval_idx]
    x_test_cat = x_cat_tokens[test_idx]
    durations_train_val = durations[trainval_idx]
    durations_test = durations[test_idx]
    events_train_val = events[trainval_idx]
    events_test = events[test_idx]

    x_train_num = x[train_idx]
    x_val_num = x[val_idx]
    x_train_cat = x_cat_tokens[train_idx]
    x_val_cat = x_cat_tokens[val_idx]
    durations_train = durations[train_idx]
    durations_val = durations[val_idx]
    events_train = events[train_idx]
    events_val = events[val_idx]

    survival_train = Surv.from_arrays(events_train, durations_train)
    survival_val = Surv.from_arrays(events_val, durations_val)
    survival_test = Surv.from_arrays(events_test, durations_test)
    survival_trainval = Surv.from_arrays(events_train_val, durations_train_val)

    return (x_train_num, x_val_num, x_test_num,
            x_train_cat, x_val_cat, x_test_cat,
            survival_train, durations_val, events_val, survival_test,
            cat_cardinalities, x_train_val_num, x_train_val_cat, durations_train_val, events_train_val)

