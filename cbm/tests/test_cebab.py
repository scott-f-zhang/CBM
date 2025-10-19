#!/usr/bin/env python3
"""CEBaB dataset test runner with flexible model selection.

This script runs experiments on the CEBaB dataset with configurable model selection,
variants, metrics, and early stopping.

Usage examples:
- python test_cebab.py --model all                    # Test all models
- python test_cebab.py --model bert --variants D^     # Test BERT on D^ variant only
- python test_cebab.py --model gpt2 --no_early_stopping  # Test GPT2 without early stopping
- python test_cebab.py --model lstm --metrics task    # Test LSTM task metrics only
"""

import os
import sys
import argparse
import pandas as pd
import torch
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Optional
from cbm import (
    get_cbm_standard,
    get_cbm_joint,
    get_cbm_LLM_mix_joint,
)
from cbm.config.defaults import RunConfig


def get_average_scores(score_list):
    if not score_list:
        return (0.0, 0.0)
    s1 = s2 = 0.0
    n = 0
    for a, b in score_list:
        s1 += a
        s2 += b
        n += 1
    return ((s1 / n * 100), (s2 / n * 100))


def get_tuple_2f_fmt(tp):
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"


# Model configurations
MODEL_CONFIGS = {
    'bert-base-uncased': {
        'display_name': 'BERT',
        'num_epochs': 1,
        'max_len': 512,
        'learning_rate': 1e-5,
        'supports_cm': True,
    },
    'roberta-base': {
        'display_name': 'RoBERTa',
        'num_epochs': 1,
        'max_len': 512,
        'learning_rate': 1e-5,
        'supports_cm': True,
    },
    'gpt2': {
        'display_name': 'GPT2',
        'num_epochs': 20,
        'max_len': 128,
        'learning_rate': 1e-5,
        'supports_cm': True,
    },
    'lstm': {
        'display_name': 'LSTM',
        'num_epochs': 20,
        'max_len': 128,
        'learning_rate': 1e-2,
        'supports_cm': False,  # LSTM doesn't support CBE-PLMs-CM
    },
}

DATASET = "cebab"
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_learning_rate(model_name: str) -> Optional[float]:
    """Get learning rate for a model."""
    return MODEL_CONFIGS.get(model_name, {}).get('learning_rate', 1e-5)


def _variant_plan_for_cebab(variants_filter: str):
    """Get variant plan for CEBaB dataset."""
    mapping = {"D": ("pure", "D"), "D^": ("aug", "D^")}
    plans = []
    if variants_filter in ("all", "D"):
        plans.append(mapping["D"])
    if variants_filter in ("all", "D^"):
        plans.append(mapping["D^"])
    return plans


def run_experiments_for_function(func_name: str, func, model_name: str, 
                                variants_filter: str, metrics_filter: str, 
                                early_stopping: bool, fasttext_path: Optional[str] = None):
    rows = []
    config = MODEL_CONFIGS[model_name]
    display_name = config['display_name']
    num_epochs = config['num_epochs']
    max_len = config['max_len']
    
    print(f"Running {func_name} ({display_name}) on CEBaB...")
    
    variant_plan = _variant_plan_for_cebab(variants_filter)
    lr = get_learning_rate(model_name)
    
    for variant, data_type in variant_plan:
        try:
            print(f"\tVariant={variant}  data_type={data_type}  lr={lr}  epochs={num_epochs}")
            
            # Prepare function arguments
            kwargs = {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'dataset': DATASET,
                'max_len': max_len,
                'batch_size': 8,
                'optimizer_lr': lr,
                'variant': variant,
                'early_stopping': early_stopping,
            }
            
            # Add fasttext_path for LSTM
            if model_name == 'lstm' and fasttext_path:
                kwargs['fasttext_path'] = fasttext_path
            
            score = func(**kwargs)
            
            # Process results based on function type
            if isinstance(score, dict):
                task_scores = score.get('task', []) if metrics_filter in ("task", "all") else []
                concept_scores = score.get('concept', []) if metrics_filter in ("concept", "all") else []
                score_out = {k: v for k, v in (("task", task_scores), ("concept", concept_scores)) if v}
            else:
                score_out = score if metrics_filter in ("task", "all") else []
                
        except Exception as e:
            print(f"\tWarning: {func_name}/{DATASET}/{model_name}/variant={variant} failed: {e}")
            print(traceback.format_exc())
            score_out = []
            
        rows.append({
            'dataset': DATASET,
            'data_type': data_type,
            'function': func_name,
            'model': model_name,
            'score': score_out if not isinstance(score_out, dict) else score_out.get('task', []),
            'concept_score': [] if not isinstance(score_out, dict) else score_out.get('concept', []),
        })
    return rows


def run_all_experiments(models_sel, variants_filter: str, metrics_filter: str,
                       early_stopping: bool, fasttext_path: Optional[str] = None) -> pd.DataFrame:
    all_rows = []
    
    for model_name in models_sel:
        config = MODEL_CONFIGS[model_name]
        supports_cm = config['supports_cm']
        
        # Define functions based on model capabilities
        funcs = {
            'PLMs': get_cbm_standard,
            'CBE-PLMs': get_cbm_joint,
        }
        
        # Add CBE-PLMs-CM only for models that support it
        if supports_cm:
            funcs['CBE-PLMs-CM'] = get_cbm_LLM_mix_joint
        
        for name, fn in funcs.items():
            all_rows.extend(run_experiments_for_function(
                name, fn, model_name, variants_filter, 
                metrics_filter, early_stopping, fasttext_path
            ))
    
    return pd.DataFrame(all_rows)


def build_pivot_table(df: pd.DataFrame, models_sel):
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    func_order = ["PLMs", "CBE-PLMs", "CBE-PLMs-CM"]
    
    # Create model order based on selected models
    model_order = [MODEL_CONFIGS[model]['display_name'] for model in models_sel]
    model_mapping = {model: MODEL_CONFIGS[model]['display_name'] for model in models_sel}

    if not df.empty:
        dfp = df.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')
        dfp = dfp.reset_index()
        dfp['model'] = dfp['model'].map(model_mapping).fillna(dfp['model'])
        dfp = dfp.set_index(['function', 'model'])
        dfp = dfp.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    return df, dfp


def main():
    parser = argparse.ArgumentParser(description="CEBaB dataset test with flexible model selection")
    parser.add_argument("--model", default="all", 
                        choices=["bert-base-uncased", "roberta-base", "gpt2", "lstm", "all"],
                        help="Model to test")
    parser.add_argument("--variants", default="all", choices=["D", "D^", "all"], 
                        help="Data variants to run")
    parser.add_argument("--metrics", default="all", choices=["task", "concept", "all"], 
                        help="Which metrics to evaluate/report")
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping (default)")
    parser.add_argument("--no_early_stopping", action="store_true", 
                        help="Disable early stopping (fixed epochs)")
    parser.add_argument("--fasttext", type=str, default=None, 
                        help="Path to FastText cc.en.300.bin; or set FASTTEXT_BIN env var")
    args = parser.parse_args()

    # Determine model selection
    if args.model == "all":
        models_sel = list(MODEL_CONFIGS.keys())
    else:
        models_sel = [args.model]
    
    variants_filter = args.variants
    metrics_filter = args.metrics
    
    # Determine early stopping setting
    early_stopping = not args.no_early_stopping
    
    # Resolve fasttext path
    fasttext_path = args.fasttext or os.environ.get("FASTTEXT_BIN")
    
    # Generate output filename
    model_suffix = args.model if args.model != "all" else "all"
    early_stopping_suffix = "no_early_stopping" if not early_stopping else "early_stopping"
    OUTPUT_CSV = os.path.join(TESTS_DIR, "test_results", f"result_cebab_{model_suffix}_{early_stopping_suffix}.csv")
    
    # Ensure test_results directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"Dataset: CEBaB")
    print(f"Early stopping: {'ENABLED' if early_stopping else 'DISABLED'}")
    print(f"Models: {models_sel}")
    print(f"Variants: {variants_filter}")
    print(f"Metrics: {metrics_filter}")
    
    df = run_all_experiments(models_sel, variants_filter, metrics_filter, 
                           early_stopping, fasttext_path)
    df, dfp = build_pivot_table(df, models_sel)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nCEBaB Results:")
    print(dfp)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
