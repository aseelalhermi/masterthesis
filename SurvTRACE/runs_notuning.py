import argparse
import wandb
import torch
import numpy as np
from copy import deepcopy

from survtrace.config import STConfig
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.dataset import load_data
from survtrace.train_utils import Trainer
from survtrace.evaluate_utils import Evaluator
from runs_main import run_optuna_then_evaluate_n_seeds
from survtrace.evaluate_utils import plot_survival_analysis

def parse_args():
    parser = argparse.ArgumentParser(description="Run SurvTrace Experiment")
    parser.add_argument("--dataset", type=str, default="metabric", 
                        choices=["metabric", "support", "flchain", "gbsg"])
    parser.add_argument("--attn", type=str, default="saint", 
                        choices=["saint", "regular"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--add_mask", action='store_true')
    parser.add_argument("--pos_enc", type=str, default="none", choices=["absolute", "learnable", "sinosoidal", "rope", "none"])

    # --- Best params override ---
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=None)
    
    parser.add_argument("--skip_optuna", action="store_true", 
                        help="Use provided hyperparameters instead of Optuna search.")
    return parser.parse_args()

def main():
    args = parse_args()
    print(1)

    # Prepare config
    config = deepcopy(STConfig)
    config.data = args.dataset
    config.add_mask = args.add_mask
    config.position_embedding_type = None if args.pos_enc == "none" else args.pos_enc
    config.use_saint_attention = args.attn == "saint"
    print(2)
    run_name = f"{args.dataset}_{args.pos_enc}_{args.attn}"
    wandb.init(project="survtrace-argparse", config=config, name=run_name)
    print(3)

    if args.skip_optuna:
        # --- Use best params directly ---
        if args.hidden_size: 
            config.hidden_size = args.hidden_size
        if args.intermediate_size:
            config.intermediate_size = args.intermediate_size
        if args.num_hidden_layers:
            config.num_hidden_layers = args.num_hidden_layers
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        if args.weight_decay:
            config.weight_decay = args.weight_decay
        if args.num_attention_heads:
            config.num_attention_heads = args.num_attention_heads
        if args.attention_probs_dropout_prob:
            config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.hidden_dropout_prob:
            config.hidden_dropout_prob = args.hidden_dropout_prob
        print(4)

        all_metrics = []
        for seed in range(args.n_seeds):
            config.seed = seed

            wandb.init(project="survtrace-optuna", config=config, name=f"seed_{seed}", reinit=True)

            set_random_seed(seed)

            (
                df_all, df_train, y_train_trans, df_test, y_test_raw, df_val, y_val_trans, df_y_val_raw,
                X_temp, y_trainval_trans, y_trainval_raw, y_temp_trans, df_trainval
            ) = load_data(config)

            trainval_set = (X_temp, y_trainval_trans)
            train_set = (df_train, y_train_trans)
            val_set   = (df_val, y_val_trans)
            test_set  = (df_test, y_test_raw)  # <-- raw for test

            model = SurvTraceSingle(config)
            trainer = Trainer(model)
            train_loss, val_loss = trainer.fit(train_set, val_set,
                        batch_size=64, epochs=100,
                        learning_rate=config.learning_rate, weight_decay=config.weight_decay)

            for epoch, (t, v) in enumerate(zip(train_loss, val_loss)):
              wandb.log({"train_loss": t, "val_loss": v, "epoch": epoch})
              
              # Re-run a forward pass with a small batch for visualization
              model.eval()
              with torch.no_grad():
                  sample_x = df_train.iloc[:8]
                  x_cat = torch.tensor(sample_x.iloc[:, :config.num_categorical_feature].values).long()
                  x_num = torch.tensor(sample_x.iloc[:, config.num_categorical_feature:].values).float()


                  if model.use_gpu:
                      x_cat = x_cat.cuda()
                      x_num = x_num.cuda()


                  outputs = model(input_ids=x_cat, input_nums=x_num)

            evaluator = Evaluator(df_all, train_index=df_train.index)
            metrics = evaluator.eval(model, test_set)

            # Only for plotting
            df_test = df_test.reset_index(drop=True)
            x_test = test_set[0]
            surv = model.predict_surv(x_test, batch_size=64)
            risk = 1 - surv
            durations_test = y_test_raw['duration'].values
            events_test = y_test_raw['event'].values

            plot_survival_analysis(
                surv=surv,
                risk=risk,
                durations_test=durations_test,
                events_test=events_test,
                times=config.duration_index[1:-1],
                horizons=config.horizons,
                metrics=metrics,
            )

            metrics = evaluator.eval(model, test_set)
            wandb.log(metrics)
            wandb.finish()

            all_metrics.append(metrics)

        # Aggregate results
        from collections import defaultdict
        agg_metrics = defaultdict(list)
        for m in all_metrics:
            for k, v in m.items():
                agg_metrics[k].append(v)

        print("\n=== Final Results Across 10 Seeds ===")
        for k, v in agg_metrics.items():
            print(f"{k} - mean: {np.mean(v):.4f}, std: {np.std(v):.4f}")

    else:
        # --- Use Optuna ---
        run_optuna_then_evaluate_n_seeds(
            n_trials=args.n_trials, n_seeds=args.n_seeds, base_config=config
        )

    wandb.finish()

if __name__ == "__main__":
    main()



