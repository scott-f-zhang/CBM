import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_len: int, variant: str = "manual", expand_concepts=None):
        # Internal loading aligned with CEBaBDataset, variants: manual / aug_manual / gen / aug_gen
        if expand_concepts is None:
            expand_concepts = (variant != "manual")

        # Resolve repo root and dataset directory for stable paths
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        IMDB_DIR = os.path.join(ROOT_DIR, "dataset", "imdb")

        def _read_with_fallback(preferred: str, alternatives):
            # Candidate search locations: imdb/ and imdb/New/
            search_dirs = [IMDB_DIR, os.path.join(IMDB_DIR, "New")]
            candidates = [preferred] + alternatives
            for d in search_dirs:
                for fname in candidates:
                    p = os.path.join(d, fname)
                    if os.path.exists(p):
                        if d != IMDB_DIR or fname != preferred:
                            print(f"[IMDBDataset] Warning: '{preferred}' not found under {IMDB_DIR}, using '{p}' instead")
                        return pd.read_csv(p)
            print(f"[IMDBDataset] Warning: none of { [preferred]+alternatives } found under {IMDB_DIR}. This split will be empty.")
            return pd.DataFrame(columns=[
                'review','sentiment','acting','storyline','emotional arousal','cinematography',
                'soundtrack','directing','background setting','editing'
            ])

        if variant == "manual":
            frames = {
                "train": _read_with_fallback("IMDB-train-manual.csv", ["IMDB-train-manual.csv", "IMDB-train-generated.csv"]),
                "val": _read_with_fallback("IMDB-dev-manual.csv", ["IMDB-dev-manual.csv", "IMDB-dev-generated.csv"]),
                "test": _read_with_fallback("IMDB-test-manual.csv", ["IMDB-test-manual.csv", "IMDB-test-generated.csv"]),
            }
        elif variant == "gen":
            frames = {
                "train": _read_with_fallback("IMDB-train-generated.csv", ["IMDB-train-generated.csv", "IMDB-train-manual.csv"]),
                "val": _read_with_fallback("IMDB-dev-generated.csv", ["IMDB-dev-generated.csv", "IMDB-dev-manual.csv"]),
                "test": _read_with_fallback("IMDB-test-generated.csv", ["IMDB-test-generated.csv", "IMDB-test-manual.csv"]),
            }
        elif variant == "aug_manual":
            # Currently mirrors manual files but enables extra noisy concepts
            frames = {
                "train": pd.read_csv(os.path.join(IMDB_DIR, "IMDB-train-manual.csv")),
                "val": pd.read_csv(os.path.join(IMDB_DIR, "IMDB-dev-manual.csv")),
                "test": pd.read_csv(os.path.join(IMDB_DIR, "IMDB-test-manual.csv")),
            }
        elif variant == "aug_gen":
            train_manual = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-train-manual.csv"))
            val_manual = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-dev-manual.csv"))
            test_manual = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-test-manual.csv"))
            train_gen = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-train-generated.csv"))
            val_gen = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-dev-generated.csv"))
            test_gen = pd.read_csv(os.path.join(IMDB_DIR, "IMDB-test-generated.csv"))
            frames = {
                "train": pd.concat([train_manual, train_gen], ignore_index=True),
                "val": pd.concat([val_manual, val_gen], ignore_index=True),
                "test": pd.concat([test_manual, test_gen], ignore_index=True),
            }
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        df = frames[split]

        # Text and label columns per run_imdb
        self.data = df
        self.text = df["review"].astype(str)
        self.labels = df["sentiment"]

        # Gold concepts (4)
        self.acting_aspect = df["acting"]
        self.storyline_aspect = df["storyline"]
        self.emotional_aspect = df["emotional arousal"]
        self.cinematography_aspect = df["cinematography"]

        # Optional noisy concepts (4)
        self.extra = None
        if expand_concepts and all(col in df for col in [
            "soundtrack", "directing", "background setting", "editing"
        ]):
            self.extra = {
                "soundtrack": df["soundtrack"],
                "directing": df["directing"],
                "background": df["background setting"],
                "editing": df["editing"],
            }

        self.map_dict = {"Negative": 0, "Negative ": 0, "Positive": 1, "unknown": 2, "Unkown": 2}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.variant = variant
        self.expand_concepts = expand_concepts
        
        # Dataset metadata
        self.final_label = ['sentiment']
        self.final_label_vals = [0, 1]
        self.concept_vals = [0, 1, 2]
        
        # Set concepts based on whether extra concepts are available
        if self.extra is not None:
            self.concepts = ['acting', 'storyline', 'emotional', 'cinematography', 
                           'soundtrack', 'directing', 'background', 'editing']
        else:
            self.concepts = ['acting', 'storyline', 'emotional', 'cinematography']

        # Keep all rows; no 'no majority' filtering used here
        self.indices = list(range(len(self.labels)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        text = self.text[i]
        label = 1 if str(self.labels[i]).strip() == "Positive" else 0

        acting_concept = self.map_dict[str(self.acting_aspect[i]).strip()]
        storyline_concept = self.map_dict[str(self.storyline_aspect[i]).strip()]
        emotional_concept = self.map_dict[str(self.emotional_aspect[i]).strip()]
        cinematography_concept = self.map_dict[str(self.cinematography_aspect[i]).strip()]

        concept_labels = [acting_concept, storyline_concept, emotional_concept, cinematography_concept]

        if self.extra is not None:
            soundtrack_concept = self.map_dict[str(self.extra["soundtrack"][i]).strip()]
            directing_concept = self.map_dict[str(self.extra["directing"][i]).strip()]
            background_concept = self.map_dict[str(self.extra["background"][i]).strip()]
            editing_concept = self.map_dict[str(self.extra["editing"][i]).strip()]
            concept_labels = concept_labels + [
                soundtrack_concept, directing_concept, background_concept, editing_concept
            ]

        enc = self.tokenizer.encode_plus(
            str(text), add_special_tokens=True, max_length=self.max_len,
            truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "acting_concept": torch.tensor(acting_concept, dtype=torch.long),
            "storyline_concept": torch.tensor(storyline_concept, dtype=torch.long),
            "emotional_concept": torch.tensor(emotional_concept, dtype=torch.long),
            "cinematography_concept": torch.tensor(cinematography_concept, dtype=torch.long),
            "concept_labels": torch.tensor(concept_labels, dtype=torch.long),
        }
        if self.extra is not None:
            item.update({
                "soundtrack_concept": torch.tensor(concept_labels[4], dtype=torch.long),
                "directing_concept": torch.tensor(concept_labels[5], dtype=torch.long),
                "background_concept": torch.tensor(concept_labels[6], dtype=torch.long),
                "editing_concept": torch.tensor(concept_labels[7], dtype=torch.long),
            })
        return item


