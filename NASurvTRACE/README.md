# Modified NA-SurvTrace 🧠⏳  
*A Transformer-Based Model for Survival Analysis*

SurvTrace is a Transformer-based deep learning model tailored for survival analysis. It incorporates both intra-sample and inter-sample attention (SAINT-style), supports multiple positional encodings, NA attention mask, and provides strong performance on clinical and tabular datasets.

---

## 🔧 Features

- ✅ Transformer-based architecture for survival prediction
- 🧠 SAINT-style **column (inter-sample)** and **row (intra-sample)** attention
- ⏱️ Support for **RoPE**, **learnable**, **sinusoidal**, and **absolute** positional encodings
- **Masking out missing values from attention calculation**
- 📉 Metrics: Integrated Brier Score (IBS), C-Index, time-dependent AUC, IPCW
- 🧪 Hyperparameter tuning with Optuna
- 📊 Attention visualization with Weights & Biases (wandb)
- 📁 Modular code structure and multiple dataset support

---

## 📁 Project Structure

```

survtrace/
├── config.py                # Configuration settings
├── dataset.py               # Dataset loading & preprocessing
├── evaluate_utils.py        # Evaluation and plotting utilities
├── losses.py                # Survival-specific losses 
├── model.py                 # Main model wrapper
├── modeling_bert.py         # Transformer + attention logic
├── train_utils.py           # Training loop and helpers
├── utils.py                 # Miscellaneous utilities

runs_main.py             # Full training + Optuna tuning
runs_notuning.py         # Train with fixed hyperparameters
run_survtrace.py        # CLI with argparse
requirements.txt         # Python dependencies
README.md

````

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
````

### 2. Run with Optuna Tuning

```bash
python run_survtrace.py \
  --dataset metabric \
  --attn saint \
  --pos_enc rope \
  --n_trials 100 \
  --n_seeds 10 \
  --output_attentions
```

### 3. Run with Fixed Parameters (No Tuning)

```bash
python runs_notuning.py \
  --dataset support \
  --attn regular \
  --pos_enc learnable \
  --skip_optuna
```

### 4. Available Arguments

| Argument              | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `--dataset`           | Dataset name: `metabric`, `support`, `flchain`, etc.   |
| `--attn`              | Attention type: `saint` or `regular`                   |
| `--pos_enc`           | Positional encoding: `rope`, `learnable`, `none`, etc. |
| `--add_mask`          | Add missingness mask (default: False)                  |
| `--n_trials`          | Optuna tuning trials (default: 100)                    |
| `--n_seeds`           | Number of seeds to average (default: 10)               |
| `--output_attentions` | Log attention maps with wandb                          |

---

## 📊 Evaluation

The model computes:

* **Integrated Brier Score (IBS)**
* **C-Index (Concordance)**
* **Time-dependent AUC**
* **IPCW-adjusted metrics**

Evaluation is done using raw durations and events, with support for stratified splitting.

---

## 📷 Visualizations

SurvTrace supports:

* 📈 Survival curves and risk predictions
* 🧠 Attention heatmaps (row-wise and column-wise)
* 📉 Training/validation loss tracking

All visualizations are logged to [Weights & Biases](https://wandb.ai).

---


## ✨ Acknowledgements

* SurvTrace https://github.com/RyanWangZf/SurvTRACE
* SAINT: Self-Attention and Intersample Attention for Tabular Data https://github.com/somepago/saint/tree/main
* Rotary Positional Embeddings (RoPE)
* scikit-survival, lifelines, pycox, Optuna, and wandb

---

## 🙋 Questions?

Feel free to open an issue or discussion on the repository.



