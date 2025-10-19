#!/usr/bin/env python3
"""Run CBM experiments across datasets and model backbones.

This script centralizes experiment configuration at the top of the file and
exposes a small set of helper functions to:
- execute runs for each pipeline family
- aggregate raw scores into a DataFrame
- build a pivoted, human-readable summary
"""
import os
import sys
import pandas as pd

# Ensure project root on sys.path so we can import the main package
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cbm import (
    get_cbm_standard,
    get_cbm_joint,
    get_cbm_LLM_mix_joint,
)
from cbm.config.defaults import RunConfig


def get_average_scores(score_list):
    """Average a list of (acc, macro_f1) tuples and scale to percentage.

    Returns a pair (acc_pct, macro_f1_pct) in [0, 100].
    """
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
    """Format a (x, y) tuple as a string with two decimals: 'xx.xx/yy.yy'."""
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"

# Base run settings aligned with main.config.defaults.RunConfig
BASE_RUN = RunConfig(
    num_epochs=20,
    max_len=512,
    batch_size=8,
)

# Which datasets and models to run
DATASETS = ["essay"]  # choose from: "cebab", "imdb", "essay"
MODELS = ["bert-base-uncased", "roberta-base", "gpt2", "lstm"]
MODELS = ["bert-base-uncased", "roberta-base"]

# Output CSV path
OUTPUT_CSV = os.path.join(MAIN_DIR, "result_essay.csv")


from typing import Optional


def get_learning_rate(model_name: str) -> Optional[float]:
    """Return a reasonable default learning rate for a given backbone name.
    
    For essay dataset, using optimized learning rates found by learning rate finder.
    Updated based on actual LR finder results with 6-class configuration.
    """
    lr_rate_dt = {
        'lstm': 5e-4,           # Essay optimized: 0.0005 (joint pipeline, combined_score=1.361) - 实际最优值
        'gpt2': 5e-5,           # Essay optimized: 5e-5 (joint pipeline, combined_score=1.470)
        'roberta-base': 2e-5,   # Essay optimized: 2e-5 (joint pipeline, combined_score=1.508)
        'bert-base-uncased': 2e-5,  # Essay optimized: 2e-5 (joint pipeline, combined_score=1.465)
    }
    return lr_rate_dt.get(model_name)


def run_experiments_for_function(func_name: str, func):
    """Run one pipeline family (e.g., PLMs) across datasets and models.

    For CEBaB, run both variants corresponding to D (pure) and D^ (aug_cebab).
    Returns a list of dict rows compatible with pandas.DataFrame.
    """
    rows = []
    print(f"Running {func_name}...")

    for dataset in DATASETS:
        print(f"\tRunning dataset: {dataset}...")

        # Determine which variants to run and the data_type label
        if dataset == 'cebab':
            variant_plan = [
                ('pure', 'D'),      # original CEBaB
                ('aug', 'D^'),      # aug_cebab
            ]
        else:
            # For IMDB, map manual->D and gen->D^
            variant_plan = [('manual', 'D'), ('gen', 'D^')]

        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... with learning rate: {lr}")

            for variant, data_type in variant_plan:
                # CBE-PLMs-CM supports LSTM, just not on D variants
                # CM skips D on both datasets (cebab pure, imdb manual)
                if func_name == 'CBE-PLMs-CM':
                    if (dataset == 'cebab' and variant == 'pure') or (dataset == 'imdb' and variant == 'manual'):
                        print("\t\tSkipping D for CBE-PLMs-CM per paper")
                        continue
                try:
                    kwargs = dict(
                        model_name=model_name,
                        num_epochs=BASE_RUN.num_epochs,
                        dataset=dataset,
                        max_len=BASE_RUN.max_len,
                        batch_size=BASE_RUN.batch_size,
                        optimizer_lr=lr,
                    )
                    if variant is not None:
                        kwargs['variant'] = variant
                    result = func(**kwargs)
                except Exception as e:
                    print(f"\t\tWarning: {func_name}/{dataset}/{model_name}/variant={variant} failed: {e}")
                    result = []

                # Normalize result shape per family
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
                    'dataset': dataset,
                    'data_type': data_type,
                    'function': func_name,
                    'model': model_name,
                    'score': task_scores,
                    'concept_score': concept_scores,
                })

    return rows


def run_all_experiments() -> pd.DataFrame:
    """Run all pipeline families and return a long-form results DataFrame."""
    plms_funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        'CBE-PLMs-CM': get_cbm_LLM_mix_joint,
    }
    all_rows = []

    for fname, f in plms_funcs.items():
        all_rows.extend(run_experiments_for_function(fname, f))

    df = pd.DataFrame(all_rows)
    return df


def build_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create pivot tables and formatted output.

    - For CEBaB, pivot by data_type -> columns D and D^.
    - For IMDB, keep a simple dataset column.
    Returns the pair (df_with_scores, pivot_df_cebab, pivot_df_imdb).
    """
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    func_order = ["PLMs", "CBE-PLMs", "CBE-PLMs-CM"]
    model_order = ["LSTM", "GPT2", "BERT", "RoBERTa"]
    mapping = {
        'lstm': 'LSTM',
        'gpt2': 'GPT2',
        'bert-base-uncased': 'BERT',
        'roberta-base': 'RoBERTa',
    }

    # Build unified multi-index columns: (dataset, D/D^, metric)
    # Prepare task long
    task_df = df.copy()
    task_df['score_avg'] = task_df['score'].apply(get_average_scores)
    task_df['fmt'] = task_df['score_avg'].apply(get_tuple_2f_fmt)
    task_df['metric'] = 'task'
    # Prepare concept long
    concept_df = df.copy()
    if 'concept_score' in concept_df.columns:
        concept_df['score_avg'] = concept_df['concept_score'].apply(get_average_scores)
        concept_df['fmt'] = concept_df['score_avg'].apply(get_tuple_2f_fmt)
        concept_df['metric'] = 'concept'
    else:
        concept_df = concept_df.iloc[0:0]

    merged = pd.concat([task_df[['function','model','dataset','data_type','metric','fmt']],
                        concept_df[['function','model','dataset','data_type','metric','fmt']]], ignore_index=True)

    merged = merged.reset_index(drop=True)
    # Normalize model display names
    merged['model'] = merged['model'].map(mapping)

    # Pivot to columns MultiIndex
    wide = merged.pivot_table(index=['function','model'],
                              columns=['dataset','data_type','metric'],
                              values='fmt', aggfunc='first')
    # Reindex rows
    wide = wide.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function","model"]))

    # Ensure column order includes all combinations
    desired_cols = []
    for ds in ['cebab','imdb']:
        for dt in ['D','D^']:
            for m in ['task','concept']:
                desired_cols.append((ds, dt, m))
    # Add missing columns with NaN
    for col in desired_cols:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[desired_cols]

    return df, wide


def main():
    """Entrypoint: run experiments, save CSV, and print CEBaB D/D^ and IMDB pivots."""
    df = run_all_experiments()
    df, dfp = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nUnified Pivot (dataset, D/D^, task/concept):")
    print(dfp)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
