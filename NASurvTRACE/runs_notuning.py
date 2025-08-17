import argparse
import wandb
from copy import deepcopy
from survtrace.config import STConfig
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.dataset import load_data
from survtrace.train_utils import Trainer
from survtrace.evaluate_utils import Evaluator
from runs_main import run_optuna_then_evaluate_n_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="Run SurvTrace Experiment")
    parser.add_argument("--dataset", type=str, default="metabric", 
                        choices=["metabric", "support", "flchain", "gbsg"])
    parser.add_argument("--pos_enc", type=str, default="rope", 
                        choices=["absolute", "learnable", "sinosoidal", "rope", "none"])
    parser.add_argument("--attn", type=str, default="saint", 
                        choices=["saint", "regular"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--output_attentions", action="store_true")
    parser.add_argument("--max_position_embeddings", type=int, default=512) 
    parser.add_argument("--add_mask", action='store_true')

    # --- Best params override ---
    parser.add_argument("--hidden_size", type=int, default=None)
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
    config.output_attentions = args.output_attentions
    config.max_position_embeddings = args.max_position_embeddings
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
            set_random_seed(seed)
            print(5)

            (
                df,
                df_train, y_train_trans, y_train_raw,
                df_test, y_test_raw,
                df_val, y_val_trans, y_val_raw,
                attention_mask, X_temp, y_trainval_trans, 
                y_trainval_raw, y_temp_trans, df_trainval
            ) = load_data(config)

            train_set = (df_train, y_train_trans, attention_mask.loc[df_train.index])
            val_set   = (df_val, y_val_trans, attention_mask.loc[df_val.index])
            test_set  = (df_test, y_test_raw, attention_mask.loc[df_test.index])

            model = SurvTraceSingle(config)
            trainer = Trainer(model)
            trainer.fit(train_set, val_set, batch_size=64, epochs=100,
                        learning_rate=config.learning_rate, weight_decay=config.weight_decay)
            print(6)
            evaluator = Evaluator(df, train_index=df_train.index)
            metrics = evaluator.eval(model, test_set)
            all_metrics.append(metrics)

        # Aggregate metrics
        from collections import defaultdict
        import numpy as np
        agg_metrics = defaultdict(list)
        for m in all_metrics:
            for k, v in m.items():
                agg_metrics[k].append(v)
        print("\n=== Final Results Across Seeds ===")
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



