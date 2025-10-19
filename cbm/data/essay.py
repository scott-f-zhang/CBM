from typing import Optional, List
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


ESSAY_CONCEPT_COLUMNS: List[str] = [
    "FC", "CC", "TU", "CP", "R", "DU", "EE", "FR"
]


class EssayDataset(Dataset):
    """Dataset wrapper for essay data prepared as CSVs with columns:

    Required columns per row:
      - text: string (already combined Q/A text)
      - label: int (0-5, representing rounded score from score_avg)
      - concept columns: FC, CC, TU, CP, R, DU, EE, FR with values in
        {"Positive", "Negative", "unknown"}

    variant: "manual" | "generated"
      manual -> train_manual.csv / dev_manual.csv / test_manual.csv
      generated -> train_generated.csv / dev_generated.csv / test_generated.csv
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        max_len: int,
        variant: str = "manual",
    ):
        assert split in ("train", "val", "test"), f"Unsupported split: {split}"
        assert variant in ("manual", "generated"), f"Unsupported variant: {variant}"

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.variant = variant

        # Resolve repo root and dataset directory to avoid CWD dependence
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        ESSAY_DIR_ROOT = os.path.join(ROOT_DIR, "dataset", "essay")
        ESSAY_DIR_CLEANED = os.path.join(ESSAY_DIR_ROOT, "cleaned")

        # Map 'val' split to files named with 'dev' prefix in cleaned outputs
        split_prefix = 'dev' if split == 'val' else split
        fname = f"{split_prefix}_{'manual' if variant == 'manual' else 'generated'}.csv"
        # Prefer cleaned/ paths as in data_prepare.ipynb; fallback to essay/
        candidate_paths = [
            os.path.join(ESSAY_DIR_CLEANED, fname),
            os.path.join(ESSAY_DIR_ROOT, fname),
        ]
        csv_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"EssayDataset cannot find file in cleaned/ or root: {candidate_paths}"
            )

        df = pd.read_csv(csv_path)

        # Mandatory columns
        required_cols = ["text", "label"] + ESSAY_CONCEPT_COLUMNS
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

        # Basic fields
        self.data = df
        self.text = df["text"].astype(str)
        self.labels = df["label"].astype(int)

        # Map concept string labels to {0,1,2}: Negative->0, Positive->1, unknown->2
        def concept_to_int(v: str) -> int:
            s = str(v).strip()
            if s == "Positive":
                return 1
            if s == "Negative":
                return 0
            return 2

        self.concepts = {}
        for c in ESSAY_CONCEPT_COLUMNS:
            self.concepts[c] = df[c].apply(concept_to_int).astype(int)

        # Keep all rows
        self.indices = list(range(len(self.labels)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        text = self.text[i]
        label = int(self.labels[i])

        # Build concept label vector in fixed order
        concept_values = [int(self.concepts[c][i]) for c in ESSAY_CONCEPT_COLUMNS]

        enc = self.tokenizer.encode_plus(
            str(text), add_special_tokens=True, max_length=self.max_len,
            truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "concept_labels": torch.tensor(concept_values, dtype=torch.long),
        }

        # Also expose per-concept fields (optional, consistent with other datasets)
        for idx, c in enumerate(ESSAY_CONCEPT_COLUMNS):
            item[f"{c}_concept"] = torch.tensor(concept_values[idx], dtype=torch.long)

        return item


