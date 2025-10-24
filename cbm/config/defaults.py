from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class RunConfig:
    mode: str = "standard"
    model_name: str = "bert-base-uncased"
    max_len: int = 128
    batch_size: int = 8
    num_epochs: int = 1
    optimizer_lr: float = 1e-5
    # 'cebab' | 'imdb' | 'essay' | 'qa'
    dataset: str = "cebab"
    # Enable early stopping during training
    early_stopping: bool = False
    # Number of epochs to wait for improvement before stopping
    early_stopping_patience: int = 5

    # unified variant:
    # cebab: 'pure'|'aug'|'aug_yelp'|'aug_both'
    # imdb: 'manual'|'aug_manual'|'gen'|'aug_gen'
    # essay, qa: single variant (no variant parameter needed)
    variant: str = "pure"


def make_run_config(
    mode: Optional[str] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    optimizer_lr: Optional[float] = None,
    dataset: Optional[str] = None,
    variant: Optional[str] = None,
    early_stopping: Optional[bool] = None,
    early_stopping_patience: Optional[int] = None,
    default_dataset: Optional[str] = None,
    default_variant: Optional[str] = None,
) -> RunConfig:
    cfg = RunConfig()
    if default_dataset is not None:
        cfg.dataset = default_dataset
    if default_variant is not None:
        cfg.variant = default_variant
    if dataset is not None:
        cfg.dataset = dataset
    if variant is not None:
        cfg.variant = variant
    if mode is not None:
        cfg.mode = mode
    if max_len is not None:
        cfg.max_len = max_len
    if batch_size is not None:
        cfg.batch_size = batch_size
    if model_name is not None:
        cfg.model_name = model_name
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    if optimizer_lr is not None:
        cfg.optimizer_lr = optimizer_lr
    if early_stopping is not None:
        cfg.early_stopping = early_stopping
    if early_stopping_patience is not None:
        cfg.early_stopping_patience = early_stopping_patience
    return cfg


# Single source of truth for FastText binary path.
# Order of resolution: explicit arg > env FASTTEXT_BIN > project default.
DEFAULT_FASTTEXT_BIN = "/scratch/fzhan113/fasttext/cc.en.300.bin"


def resolve_fasttext_path(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        print(f"Using explicit FastText binary path: {explicit_path}")
        return explicit_path
    env_path = os.environ.get("FASTTEXT_BIN")
    if env_path:
        print(f"Using env FASTTEXT_BIN: {env_path}")
        return env_path
    print(f"Using default FastText binary path: {DEFAULT_FASTTEXT_BIN}")
    return DEFAULT_FASTTEXT_BIN
