# 🧠 SurvBench: Transformer-Based Survival Modeling

This repository provides a unified collection of models and benchmarks for tabular survival analysis, including:

- ✅ Baseline survival models (Cox, RSF, DeepSurv, DeepHit, Transformer variants)
- ✅ Modified SurvTrace 
- ✅ Transformer-based survival models with SurvSAINT
- ✅ NA attention mask with NASurvTRACE

---

## 📁 Project Structure

| Folder         | Description                                                    |
|----------------|----------------------------------------------------------------|
| `baselines/`   | Standard survival models for comparison                        |
| `survtrace/`   | SurvTRACE enhancements                                         |
| `survsaint/`   | SurvSAINT models with attention variants, pretraining, masking |
| `nasurvtrace/` | NA attention mask + SurvTRACE enhancements                     |

---

## 📄 Detailed Documentation

Please refer to the individual `README.md` files inside each subdirectory for instructions on:

- Installation
- Running experiments
- Hyperparameter tuning
- Supported models and datasets
- Logging and evaluation

---

## 🧪 Example Use Cases

- `baselines/main_final.py` → Train standard models
- `survtrace/run_experiments_final.py` → Run Optuna + SurvTRACE
- `survsaint/train_survsaint.py` → Pretraining + Survival + Attention + Time modeling
- `nasurvtrace/run_survtrace.py` → NA-attention mask SurvTRACE training

---


## 🔗 Citation

If you use this work, please cite the appropriate model-specific papers and this repository.
