from typing import List, Tuple
import numpy as np
from sklearn.metrics import f1_score


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).sum()) / float(len(labels)) if len(labels) > 0 else 0.0


def compute_macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    unique = np.unique(labels)
    scores: List[float] = []
    for u in unique:
        pred = predictions == u
        true = labels == u
        scores.append(f1_score(true, pred, average='macro'))
    return float(np.mean(scores)) if scores else 0.0


def to_numpy_concat(existing: np.ndarray, arr) -> np.ndarray:
    if existing.size == 0:
        return np.array(arr)
    return np.append(existing, np.array(arr))
