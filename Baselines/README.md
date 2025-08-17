# 🔬 Baselines

This repository provides a unified pipeline for running **baseline survival models** across tabular datasets, including both traditional and transformer-based architectures.

---

## 📁 Directory Overview

```

baselines/
├── custom_models.py          # Transformer, DeepHitTransformer, DeepSurv, etc.
├── data_loader.py            # Data loading, preprocessing, tokenization
├── train_test.py            # Training loop for standard models
├── train_test_tokenized.py  # Training loop for tokenized inputs (for Transformers)
├── main_final.py            # 🔁 CLI for training, evaluation, and Optuna tuning

````

---

## 🚀 Getting Started

### 🔧 Install dependencies

```bash
pip install -r requirements.txt
````

---

## ▶️ Run Experiments

Run survival models with multiple trials (Optuna) and multiple seeds:

```bash
python main_final.py \
  --dataset metabric \
  --models deepsurv deephit transformer deephit_transformer \
  --n_trials 100 \
  --n_runs 10
```

---

## ⚙️ Command Line Options

| Argument      | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `--dataset`   | Dataset to use: `metabric`, `support`, `gbsg`, `flchain`, etc.   |
| `--models`    | List of models to train. See options below                       |
| `--n_trials`  | Number of Optuna trials per model                                |
| `--n_runs`    | Number of seeds to run for final evaluation                      |

---

## 📚 Supported Models

| Model Name            | Description                                    |
| --------------------- | ---------------------------------------------- |
| `cox`                 | Classic Cox Proportional Hazards               |
| `rsf`                 | Random Survival Forests                        |
| `gb`                  | Gradient Boosting for survival                 |
| `deepsurv`            | Deep learning version of CoxPH                 |
| `deephit`             | Deep discrete-time survival model              |
| `transformer`         | Transformer-based Cox model                    |
| `deephit_transformer` | Transformer with DeepHit-style survival output |

---

## 📈 Evaluation Metrics

Each model is evaluated using:

* **C-Index**
* **Integrated Brier Score (IBS)**
* **Time-dependent AUC**
* **Time-dependent Brier Score**
* **Time-dependent C-Index**

Results are averaged over `n_runs` seeds and stored per dataset and model and logged to [Weights & Biases](https://wandb.ai)

---

## 🧪 Hyperparameter Optimization

Hyperparameters are tuned using **Optuna** per model per dataset using `n_trials` runs.

---

## 🙋 Acknowledgments

This framework is part of the broader **SurvSAINT** and **Modified SurvTrace** ecosystem for benchmarking survival models on tabular data. 
Wandb, Optuna, pycox, scikit-survival
