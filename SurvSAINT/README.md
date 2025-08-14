# SurvSAINT 🧠📊  
*A Self-Attention and Intersample Attention Transformer for Survival Analysis*

**SurvSAINT** is a robust transformer-based survival modeling framework that extends SAINT with support for:
- Classical Cox-based survival loss
- DeepHit loss (discrete-time modeling)
- Time-aware embeddings (via learnable or sinusoidal time encodings)
- Pretraining strategies and tabular augmentations

---

## 🔧 Key Features

- ✅ **SAINT-style Attention**: Supports `colrow`, `row`, and `col` attention styles.
- ⏱️ **Survival Losses**: Cox Proportional Hazards and DeepHit
- 🧪 **Pretraining**: Self-supervised objectives for tabular feature recovery
- ⌛ **Time Embeddings**: For modeling time-aware representations (for `--task time`)
- 🧩 **Augmentations**: CutMix, MixUp, and other augmentations for tabular data
- 🔍 **Optuna** hyperparameter tuning
- 🧠 **WandB Logging**: Full integration for tracking metrics and attention

---

## 📁 Directory Structure

```

survsaint/
├── augmentations.py           # MixUp, CutMix, Noise-based augmentations
├── datasets.py                # Dataset loading and formatting
├── losses.py                  # Cox, DeepHit, and contrastive losses
├── pretraining.py             # Pretraining objectives
├── data_openml.py             # OpenML integration and data cleaning
├── utils.py                   # Helper utilities and config handlers
├── train_survsaint.py         # 🔁 CLI: main training and Optuna entrypoint
├── models/
│   ├── **init**.py
│   ├── model.py               # Main SAINT architecture
│   ├── timemodel.py           # SurvSAINT with time-aware embeddings
│   ├── pretrainmodel.py       # Encoder for tabular pretraining
│   ├── pretrainmodel_vision.py # Optional vision encoder (if present)

````

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

---

## ⚙️ Training Commands

### ▶️ SurvSAINT (Standard Survival)

```bash
python train_survsaint.py \
  --survival \
  --active_log \
  --task survival \
  --optuna \
  --pretrain \
  --dataset metabric \
  --attentiontype colrow
```

### ▶️ SurvSAINT with DeepHit

```bash
python train_survsaint.py \
  --survival \
  --active_log \
  --task deephit \
  --optuna \
  --pretrain \
  --dataset metabric \
  --attentiontype colrow
```

### ▶️ SurvSAINT with Time Embeddings

```bash
python train_survsaint.py \
  --survival \
  --active_log \
  --task time \
  --optuna \
  --pretrain \
  --dataset metabric \
  --attentiontype colrow \
  --time_embedding_type learnable
```

---

## 🧠 Arguments (Key)

| Argument                | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `--survival`            | Enable survival loss modeling                      |
| `--task`                | Choose from `survival`, `deephit`, or `time`       |
| `--optuna`              | Enable Optuna for hyperparameter tuning            |
| `--pretrain`            | Use self-supervised pretraining                    |
| `--attentiontype`       | Attention style: `colrow`, `row`, `col`            |
| `--dataset`             | Choose from `metabric`, `support`, `gbsg`, etc.    |
| `--time_embedding_type` | For `--task time`: use `learnable` or `sinusoidal` |
| `--active_log`          | Enable Weights & Biases logging                    |

---

## 📉 Evaluation Metrics

* **C-Index**
* **Integrated Brier Score (IBS)**
* **Time-dependent Concordance**
* **Time-dependent AUC**
* **Calibration & Survival Curves**

Visual outputs and metrics are logged to [wandb.ai](https://wandb.ai).

---

## 🧪 Pretraining Objectives

Supports masked feature prediction, contrastive learning, and mixup-style denoising strategies prior to survival fine-tuning.

To disable pretraining, simply omit the `--pretrain` flag.

---

## 📊 Example Results (from CLI)

SurvSAINT outputs after each seed run (aggregated across Optuna tuning), including:

* Evaluation Metrics

---

## 🪄 Example Dataset Support

* `metabric`
* `support`
* `gbsg`
* `flchain`
---


## 🙋 Contact

For issues, questions, or ideas, feel free to open a GitHub issue or discussion.



## Acknowledgements

We would like to thank the following public repo from which we borrowed various utilites.
- https://github.com/lucidrains/tab-transformer-pytorch
- https://github.com/somepago/saint/tree/main

## **License**
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.




