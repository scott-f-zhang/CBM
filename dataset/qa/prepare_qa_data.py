#!/usr/bin/env python3
"""Data preparation script for QA dataset - PLMs and CBE-PLMs only.

This script processes the raw QA data and creates the required CSV files
for PLMs (standard) and CBE-PLMs (joint) experiments.
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
WORK_DIR = Path(__file__).parent.absolute()
DATA_DIR = WORK_DIR / "raw"
OUTPUT_DIR = WORK_DIR / "cleaned"

# Input file
SRC = DATA_DIR / "QA_train_annotated.csv"

# Output files
OUT_TRAIN = OUTPUT_DIR / "train.csv"
OUT_DEV = OUTPUT_DIR / "dev.csv"
OUT_TEST = OUTPUT_DIR / "test.csv"

# Concept columns
CONCEPT_COLS = ["FC", "CC", "TU", "CP", "R", "DU", "EE", "FR"]


def map_concept(v):
    """Map QA concept scores from 1,2,3 to 0,1,2."""
    try:
        v = int(v)
        if v == 1: return 0    # Low score → 0
        if v == 2: return 1    # Medium score → 1  
        if v == 3: return 2    # High score → 2
        return 0               # Invalid → Low
    except Exception:
        return 0               # Invalid → Low


def to_text(row):
    """Convert question and answer to text format."""
    q = str(row.get("question", "")).strip()
    a = str(row.get("student_answer", "")).strip()
    if q and a:
        return f"Q: {q}\nA: {a}"
    return a or q


def to_label_multiclass(v):
    """Convert score_avg (0-5 range) to 6-class label by rounding to nearest integer."""
    try:
        s = float(v)
        # Round to nearest integer and clip to [0, 5] for 6-class classification
        return int(round(max(0.0, min(5.0, s))))
    except Exception:
        return 0  # Default to 0 for invalid scores


def load_and_transform():
    """Load and transform the raw data."""
    print(f"Loading data from: {SRC}")
    df = pd.read_csv(SRC)
    
    out = pd.DataFrame()
    out["text"] = df.apply(to_text, axis=1)
    out["label"] = df["score_avg"].apply(to_label_multiclass)
    
    # Map concept columns
    for c in CONCEPT_COLS:
        if c in df.columns:
            out[c] = df[c].apply(map_concept)
        else:
            out[c] = "unknown"
    
    # Clean data
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    
    print(f"Loaded {len(out)} samples after cleaning")
    return out


def stratified_split(df, seed=42):
    """Perform stratified split to maintain label distribution (7:2:1 ratio)."""
    # First split: 70% train, 30% temp (dev + test)
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    # Second split: from 30%, get 20% dev and 10% test
    # 20% of total = 30% * (2/3), 10% of total = 30% * (1/3)
    dev, test = train_test_split(
        temp, test_size=1/3, stratify=temp["label"], random_state=seed
    )
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def validate_data(df, name):
    """Validate data quality."""
    print(f"\n=== {name} Validation ===")
    print(f"Shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts(normalize=True).to_dict()}")
    
    # Check concept distributions
    for col in CONCEPT_COLS:
        dist = df[col].value_counts(normalize=True).to_dict()
        print(f"{col}: {dist}")
    
    # Check for missing values
    missing = df[["text", "label"] + CONCEPT_COLS].isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values: {missing[missing > 0].to_dict()}")
    else:
        print("No missing values")


def main():
    """Main data preparation function."""
    print("=" * 60)
    print("QA DATASET PREPARATION - PLMs & CBE-PLMs ONLY")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and transform data
    df = load_and_transform()
    
    # Split data
    print("\nPerforming stratified split...")
    train, dev, test = stratified_split(df)
    
    # Save files
    print("\nSaving files...")
    train.to_csv(OUT_TRAIN, index=False)
    dev.to_csv(OUT_DEV, index=False)
    test.to_csv(OUT_TEST, index=False)
    
    print(f"Saved to: {OUTPUT_DIR}")
    
    # Validate outputs
    validate_data(train, "Train")
    validate_data(dev, "Dev")
    validate_data(test, "Test")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
def stats(name, part):
    print(f"{name}: n={len(part)}")
    label_dist = part['label'].value_counts().sort_index()
    print(f"  Label distribution: {dict(label_dist)}")
    
    stats("Train", train)
    stats("Dev", dev)
    stats("Test", test)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print("Files created:")
    print(f"  - {OUT_TRAIN}")
    print(f"  - {OUT_DEV}")
    print(f"  - {OUT_TEST}")
    print("\nReady for PLMs and CBE-PLMs experiments!")


if __name__ == "__main__":
    main()
