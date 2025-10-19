from typing import Callable, Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from ..evaluation.metrics import compute_accuracy, compute_macro_f1, to_numpy_concat


def train_one_epoch(
    model,
    head,
    data_loader: DataLoader,
    device,
    criterion,
    optimizer,
    is_lstm: bool,
) -> None:
    model.train()
    head.train()
    for batch in tqdm(data_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if is_lstm:
            # LSTM encoder already returns a fixed-size representation
            pooled = outputs
        else:
            pooled = outputs.last_hidden_state.mean(1)
        logits = head(pooled)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(
    model,
    head,
    data_loader: DataLoader,
    device,
    is_lstm: bool,
) -> Tuple[float, float]:
    model.eval()
    head.eval()
    predict_labels = np.array([])
    true_labels = np.array([])
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if is_lstm:
                pooled = outputs
            else:
                pooled = outputs.last_hidden_state.mean(1)
            logits = head(pooled)
            preds = torch.argmax(logits, dim=1)
            predict_labels = to_numpy_concat(predict_labels, preds.cpu().numpy())
            true_labels = to_numpy_concat(true_labels, labels.cpu().numpy())
    acc = compute_accuracy(predict_labels, true_labels)
    f1 = compute_macro_f1(predict_labels, true_labels)
    return acc, f1


def test_loop(
    model,
    head,
    data_loader: DataLoader,
    device,
    is_lstm: bool,
) -> List[Tuple[float, float]]:
    scores: List[Tuple[float, float]] = []
    acc, f1 = evaluate(model, head, data_loader, device, is_lstm)
    scores.append((acc, f1))
    return scores
