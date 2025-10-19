from typing import Dict, Optional
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class CEBaBDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_len: int, variant: str = "pure", expand_concepts: Optional[bool] = None):
        # Load data internally based on variant and split; manage concept expansion via flag
        # variant: 'pure' | 'aug' | 'aug_yelp' | 'aug_both'
        if expand_concepts is None:
            expand_concepts = (variant != "pure")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.variant = variant
        self.expand_concepts = expand_concepts

        # Resolve repo root and dataset directory to avoid CWD-dependent relative paths
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        CEBAB_DIR = os.path.join(ROOT_DIR, "dataset", "cebab")

        if variant == "pure":
            ds = load_dataset("CEBaB/CEBaB")
            split_map = {"train": "train_exclusive", "val": "validation", "test": "test"}
            selected = ds[split_map[split]]
            self.data = selected
        elif variant in ("aug", "aug_yelp", "aug_both"):
            if variant == "aug":
                frames = {
                    "train": pd.read_csv(os.path.join(CEBAB_DIR, "train_cebab_new_concept_single.csv")),
                    "val": pd.read_csv(os.path.join(CEBAB_DIR, "dev_cebab_new_concept_single.csv")),
                    "test": pd.read_csv(os.path.join(CEBAB_DIR, "test_cebab_new_concept_single.csv")),
                }
            elif variant == "aug_yelp":
                frames = {
                    "train": pd.read_csv(os.path.join(CEBAB_DIR, "train_yelp_exclusive_new_concept_single.csv")),
                    "val": pd.read_csv(os.path.join(CEBAB_DIR, "dev_yelp_new_concept_single.csv")),
                    "test": pd.read_csv(os.path.join(CEBAB_DIR, "test_yelp_new_concept_single.csv")),
                }
            else:  # aug_both
                train_cebab = pd.read_csv(os.path.join(CEBAB_DIR, "train_cebab_new_concept_single.csv"))
                val_cebab = pd.read_csv(os.path.join(CEBAB_DIR, "dev_cebab_new_concept_single.csv"))
                test_cebab = pd.read_csv(os.path.join(CEBAB_DIR, "test_cebab_new_concept_single.csv"))
                train_yelp = pd.read_csv(os.path.join(CEBAB_DIR, "train_yelp_exclusive_new_concept_single.csv"))
                val_yelp = pd.read_csv(os.path.join(CEBAB_DIR, "dev_yelp_new_concept_single.csv"))
                test_yelp = pd.read_csv(os.path.join(CEBAB_DIR, "test_yelp_new_concept_single.csv"))
                frames = {
                    "train": pd.concat([train_cebab, train_yelp], ignore_index=True),
                    "val": pd.concat([val_cebab, val_yelp], ignore_index=True),
                    "test": pd.concat([test_cebab, test_yelp], ignore_index=True),
                }
            self.data = frames[split]
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        split_df = self.data
        self.text = split_df["description"]
        self.labels = split_df["review_majority"]

        self.food_aspect = split_df["food_aspect_majority"]
        self.ambiance_aspect = split_df["ambiance_aspect_majority"]
        self.service_aspect = split_df["service_aspect_majority"]
        self.noise_aspect = split_df["noise_aspect_majority"]

        # Optional extra aspects for expansion if present
        self.extra = None
        if self.expand_concepts and all(col in split_df for col in [
            "cleanliness", "price", "location", "menu variety", "waiting time", "waiting area"
        ]):
            self.extra = {
                "cleanliness": split_df["cleanliness"],
                "price": split_df["price"],
                "location": split_df["location"],
                "menu_variety": split_df["menu variety"],
                "waiting_time": split_df["waiting time"],
                "waiting_area": split_df["waiting area"],
            }

        self.map_dict = {"Negative": 0, "Positive": 1, "unknown": 2, "": 2, "no majority": 2}

        # skip rows with 'no majority' labels
        self.indices = [i for i, label in enumerate(self.labels) if label != "no majority"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        text = self.text[idx]
        label = int(self.labels[idx]) - 1

        food_concept = self.map_dict[self.food_aspect[idx]]
        ambiance_concept = self.map_dict[self.ambiance_aspect[idx]]
        service_concept = self.map_dict[self.service_aspect[idx]]
        noise_concept = self.map_dict[self.noise_aspect[idx]]

        # base 4 concepts
        concept_labels = [food_concept, ambiance_concept, service_concept, noise_concept]

        # expand to 10 when configured and data available
        if self.extra is not None:
            cleanliness_concept = self.map_dict[self.extra["cleanliness"][idx]]
            price_concept = self.map_dict[self.extra["price"][idx]]
            location_concept = self.map_dict[self.extra["location"][idx]]
            menu_variety_concept = self.map_dict[self.extra["menu_variety"][idx]]
            waiting_time_concept = self.map_dict[self.extra["waiting_time"][idx]]
            waiting_area_concept = self.map_dict[self.extra["waiting_area"][idx]]
            concept_labels = concept_labels + [
                cleanliness_concept, price_concept, location_concept, menu_variety_concept,
                waiting_time_concept, waiting_area_concept,
            ]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "food_concept": torch.tensor(food_concept, dtype=torch.long),
            "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
            "service_concept": torch.tensor(service_concept, dtype=torch.long),
            "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
            "concept_labels": torch.tensor(concept_labels, dtype=torch.long),
        }
        if self.extra is not None:
            item.update({
                "cleanliness_concept": torch.tensor(concept_labels[4], dtype=torch.long),
                "price_concept": torch.tensor(concept_labels[5], dtype=torch.long),
                "location_concept": torch.tensor(concept_labels[6], dtype=torch.long),
                "menu_variety_concept": torch.tensor(concept_labels[7], dtype=torch.long),
                "waiting_time_concept": torch.tensor(concept_labels[8], dtype=torch.long),
                "waiting_area_concept": torch.tensor(concept_labels[9], dtype=torch.long),
            })
        return item
