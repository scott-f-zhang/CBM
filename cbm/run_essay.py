#!/usr/bin/env python3
"""Run complete essay dataset experiments with optimal learning rates.

This script provides a user-friendly interface to run essay dataset experiments
using the optimal learning rates found by the learning rate finder on the 7:2:1 data split.

Usage:
    python run_essay.py                    # Default: no early stopping
    python run_essay.py --early-stopping   # Enable early stopping
    python run_essay.py --no-early-stopping # Explicitly disable early stopping
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Ensure project root on sys.path (parent of the 'main' package directory)
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MAIN_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import pipelines directly to avoid circular imports with cbm/main.py
from cbm.pipelines.standard import get_cbm_standard  # no concept, baseline
from cbm.pipelines.joint import get_cbm_joint        # with concept, human annotated
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


# Learning rate configuration
LR_TYPE = "dataset_optimal"  # Options: "dataset_optimal" or "universal"

DATASET = "essay"
MODELS = ["bert-base-uncased", "roberta-base", "gpt2", "lstm"]


def get_learning_rate(model_name: str, lr_type: str = "dataset_optimal"):
    """Get learning rate for a model with option to switch between different settings."""
    
    if lr_type == "dataset_optimal":
        return {
            'lstm': 5e-4,           # Joint optimal: 5e-4 (from LR finder)
            'gpt2': 5e-5,           # Joint optimal: 5e-5 (from LR finder)
            'roberta-base': 2e-5,   # Joint optimal: 2e-5 (from LR finder)
            'bert-base-uncased': 2e-5,  # Joint optimal: 2e-5 (from LR finder)
        }.get(model_name, 1e-5)
    
    elif lr_type == "universal":
        return {
            'lstm': 1e-2,           # Universal: 1e-2 (consistent with run_cebab)
            'gpt2': 1e-4,           # Universal: 1e-4 (consistent with run_cebab)
            'roberta-base': 1e-5,   # Universal: 1e-5 (consistent with run_cebab)
            'bert-base-uncased': 1e-5,  # Universal: 1e-5 (consistent with run_cebab)
        }.get(model_name, 1e-5)
    
    else:
        raise ValueError(f"Unknown lr_type: {lr_type}. Use 'dataset_optimal' or 'universal'")


def run_experiments_for_function(func_name: str, func, early_stopping: bool, num_epochs: int):
    rows = []
    early_stopping_str = "WITH" if early_stopping else "WITHOUT"
    print(f"Running {func_name} ({early_stopping_str} EARLY STOPPING)...")

    # essay: only use manual variant (D) for both PLMs and CBE-PLMs
    variant_plan = [("manual", "D")]

    for model_name in MODELS:
        lr = get_learning_rate(model_name, LR_TYPE)
        print(f"\tRunning {model_name}... with learning rate: {lr} ({LR_TYPE})")

        for variant, data_type in variant_plan:
            try:
                kwargs = dict(
                    model_name=model_name,
                    num_epochs=num_epochs,
                    dataset=DATASET,
                    max_len=512,
                    batch_size=8,
                    optimizer_lr=lr,
                    early_stopping=early_stopping,
                )
                if variant is not None:
                    kwargs['variant'] = variant
                result = func(**kwargs)
            except Exception as e:
                print(f"\t\tWarning: {func_name}/{DATASET}/{model_name}/variant={variant} failed: {e}")
                result = []

            if func_name == 'PLMs':
                task_scores = result if isinstance(result, list) else []
                concept_scores = []
            elif func_name == 'CBE-PLMs':
                task_scores = result.get('task', []) if isinstance(result, dict) else []
                concept_scores = result.get('concept', []) if isinstance(result, dict) else []
            else:  # CBE-PLMs-CM
                if isinstance(result, dict):
                    task_scores = result.get('task', [])
                    concept_scores = result.get('concept', [])
                else:
                    task_scores = []
                    concept_scores = []

            rows.append({
                'dataset': DATASET,
                'data_type': data_type,
                'function': func_name,
                'model': model_name,
                'score': task_scores,
                'concept_score': concept_scores,
            })

    return rows


def run_all_experiments(early_stopping: bool, num_epochs: int) -> pd.DataFrame:
    plms_funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
    }
    all_rows = []
    for fname, f in plms_funcs.items():
        all_rows.extend(run_experiments_for_function(fname, f, early_stopping, num_epochs))
    return pd.DataFrame(all_rows)


def build_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    func_order = ["PLMs", "CBE-PLMs"]
    model_order = ["BERT", "RoBERTa", "GPT2", "LSTM"]
    mapping = {
        'lstm': 'LSTM',
        'gpt2': 'GPT2',
        'bert-base-uncased': 'BERT',
        'roberta-base': 'RoBERTa',
    }

    # Prepare task long
    task_df = df.copy()
    task_df['score_avg'] = task_df['score'].apply(get_average_scores)
    task_df['fmt'] = task_df['score_avg'].apply(get_tuple_2f_fmt)
    task_df['metric'] = 'task'
    # Prepare concept long
    concept_df = df.copy()
    concept_df['score_avg'] = concept_df['concept_score'].apply(get_average_scores)
    concept_df['fmt'] = concept_df['score_avg'].apply(get_tuple_2f_fmt)
    concept_df['metric'] = 'concept'

    merged = pd.concat([task_df[['function','model','dataset','data_type','metric','fmt']],
                        concept_df[['function','model','dataset','data_type','metric','fmt']]], ignore_index=True)

    merged = merged.reset_index(drop=True)
    merged['model'] = merged['model'].map(mapping)

    wide = merged.pivot_table(index=['function','model'],
                              columns=['dataset','data_type','metric'],
                              values='fmt', aggfunc='first')
    wide = wide.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function","model"]))

    # Ensure full set of columns for essay (only D variant needed)
    desired_cols = []
    for dt in ['D']:  # Only D variant needed
        for m in ['task','concept']:
            desired_cols.append((DATASET, dt, m))
    for col in desired_cols:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[desired_cols]
    return df, wide


def run_experiments(use_early_stopping=False):
    """Run essay dataset experiments with optimal learning rates.
    
    Args:
        use_early_stopping (bool): Whether to use early stopping (default: False)
    """
    
    print("=" * 80)
    print("ESSAY DATASET EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data split: 7:2:1 (train:dev:test)")
    print(f"Models: BERT, RoBERTa, GPT2, LSTM")
    print(f"Pipelines: PLMs (Standard), CBE-PLMs (Joint)")
    print(f"Early stopping: {'Enabled' if use_early_stopping else 'Disabled (Default)'}")
    print(f"Using learning rate type: {LR_TYPE}")
    print("=" * 80)
    
    try:
        print("\n🚀 Starting essay experiments...")
        
        # Determine number of epochs
        num_epochs = 20  # Default max epochs
        
        # Run experiments
        df = run_all_experiments(use_early_stopping, num_epochs)
        df, dfp = build_pivot_table(df)
        
        # Save results
        TESTS_DIR = os.path.join(MAIN_DIR, "tests")
        early_stopping_suffix = "no_early_stopping" if not use_early_stopping else "early_stopping"
        OUTPUT_CSV = os.path.join(TESTS_DIR, "test_results", f"result_essay_{early_stopping_suffix}_{LR_TYPE}.csv")
        
        # Ensure test_results directory exists
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("\nUnified Pivot (dataset, D/D^, task/concept):")
        print(dfp)
        print(f"\nSaved results to: {OUTPUT_CSV}")
        
        print("✅ Experiments completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run essay dataset experiments with configurable early stopping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_essay.py                    # Default: no early stopping (recommended)
  python run_essay.py --early-stopping   # Enable early stopping
  python run_essay.py --no-early-stopping # Explicitly disable early stopping
        """
    )
    
    # Early stopping options (mutually exclusive)
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        '--early-stopping', 
        action='store_true',
        help='Enable early stopping (patience=5 epochs)'
    )
    early_stopping_group.add_argument(
        '--no-early-stopping', 
        action='store_true',
        help='Explicitly disable early stopping (default behavior)'
    )
    
    args = parser.parse_args()
    
    # Determine early stopping setting
    use_early_stopping = args.early_stopping
    
    success = run_experiments(use_early_stopping=use_early_stopping)
    
    if success:
        print("\n🎉 All experiments completed successfully!")
        early_stopping_suffix = "no_early_stopping" if not use_early_stopping else "early_stopping"
        print(f"📁 Results saved to: tests/test_results/result_essay_{early_stopping_suffix}_{LR_TYPE}.csv")
        print("📊 Analysis summary: tests/test_results/lr_analysis_summary.md")
        print("📊 Early stopping analysis: tests/test_results/early_stopping_analysis.md")
    else:
        print("\n💥 Experiments failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
