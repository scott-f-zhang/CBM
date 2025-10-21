#!/usr/bin/env python3
"""Data preparation script for essay dataset - PLMs and CBE-PLMs only.

This script processes the raw essay data and creates the required CSV files
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
SRC = DATA_DIR / "essay.csv"

# Output files
OUT_TRAIN = OUTPUT_DIR / "train.csv"
OUT_DEV = OUTPUT_DIR / "dev.csv"
OUT_TEST = OUTPUT_DIR / "test.csv"

# Concept columns for essay data
CONCEPT_COLS = ["TC", "UE", "OC", "GM", "VA", "SV", "CTD", "FR"]


def map_concept(v):
    """Map Essay concept scores from 1,2,3,4,5 to 0,1,2,3,4."""
    try:
        v = int(v)
        if v == 1: return 0    # Lowest score → 0
        if v == 2: return 1    # Low score → 1
        if v == 3: return 2    # Medium score → 2
        if v == 4: return 3    # High score → 3
        if v == 5: return 4    # Highest score → 4
        return 2               # Invalid → Unknown
    except Exception:
        return 2               # Invalid → Unknown


def to_text(row):
    """Use full_text directly as the text content."""
    return str(row.get("full_text", "")).strip()


def to_label_multiclass(v):
    """Convert Essay score from 1,2,3,4,5,6 to 0,1,2,3,4,5 for 6-class classification."""
    try:
        s = int(v)
        if s == 1: return 0    # Lowest score → 0
        if s == 2: return 1    # Low score → 1
        if s == 3: return 2    # Medium score → 2
        if s == 4: return 3    # High score → 3
        if s == 5: return 4    # Higher score → 4
        if s == 6: return 5    # Highest score → 5
        return 0               # Invalid → Default to 0
    except Exception:
        return 0  # Default to 0 for invalid scores


def load_and_transform():
    """Load and transform the raw data."""
    print(f"Loading data from: {SRC}")
    df = pd.read_csv(SRC)
    
    out = pd.DataFrame()
    out["text"] = df.apply(to_text, axis=1)
    out["label"] = df["score"].apply(to_label_multiclass)
    
    # Map concept columns
    for c in CONCEPT_COLS:
        if c in df.columns:
            out[c] = df[c].apply(map_concept)
        else:
            out[c] = 2  # Default to unknown for missing columns
    
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
    print("ESSAY DATASET PREPARATION - PLMs & CBE-PLMs ONLY")
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
        print(f"{name}: n={len(part)}, label=1比例={part['label'].mean():.3f}")
    
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
