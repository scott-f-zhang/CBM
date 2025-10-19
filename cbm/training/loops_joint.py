from typing import Tuple, Dict
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_epoch_joint(model, head, data_loader: DataLoader, device, loss_fn, lambda_XtoC: float, is_lstm: bool) -> None:
    model.train()
    head.train()
    for batch in tqdm(data_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        concept_labels = batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels).contiguous().view(-1)

        optimizer = loss_fn["optimizer"]
        ce = loss_fn["criterion"]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if is_lstm:
            pooled_output = outputs
        else:
            pooled_output = outputs.last_hidden_state.mean(1)
        outputs2 = head(pooled_output)
        XtoC_output = outputs2[1:]
        XtoY_output = outputs2[0:1]
        # ori implementation (kept for reference): apply Sigmoid before CrossEntropy on concept logits
        # XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_output, dim=0))
        # XtoC_loss = ce(XtoC_logits, concept_labels)

        # Active implementation: for multi-class CE, use raw logits directly
        XtoC_logits = torch.cat(XtoC_output, dim=0)
        XtoC_loss = ce(XtoC_logits, concept_labels)
        XtoY_loss = ce(XtoY_output[0], label)
        loss = XtoC_loss * lambda_XtoC + XtoY_loss
        loss.backward()
        optimizer.step()


def eval_epoch_joint(model, head, data_loader: DataLoader, device, is_lstm: bool) -> Dict[str, float]:
    model.eval()
    head.eval()
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels).contiguous().view(-1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if is_lstm:
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)
            outputs2 = head(pooled_output)
            XtoC_output = outputs2[1:]
            XtoY_output = outputs2[0:1]

            predictions = torch.argmax(XtoY_output[0], axis=1)
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())

            XtoC_logits = torch.cat(XtoC_output, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())

    val_acc = float((predict_labels == true_labels).sum()) / float(len(true_labels)) if len(true_labels) else 0.0
    if len(true_labels):
        labels_unique = np.unique(true_labels)
        macro_f1_scores = []
        for lbl in labels_unique:
            macro_f1_scores.append(f1_score(true_labels == lbl, predict_labels == lbl, average='macro'))
        val_macro_f1 = float(np.mean(macro_f1_scores))
    else:
        val_macro_f1 = 0.0

    concept_acc = float((concept_predict_labels == concept_true_labels).sum()) / float(len(concept_true_labels)) if len(concept_true_labels) else 0.0
    if len(concept_true_labels):
        concept_unique = np.unique(concept_true_labels)
        concept_macro_f1_scores = []
        for c in concept_unique:
            concept_macro_f1_scores.append(f1_score(concept_true_labels == c, concept_predict_labels == c, average='macro'))
        concept_macro_f1 = float(np.mean(concept_macro_f1_scores))
    else:
        concept_macro_f1 = 0.0

    return {
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "concept_acc": concept_acc,
        "concept_macro_f1": concept_macro_f1,
    }


def test_epoch_joint(model, head, data_loader: DataLoader, device, is_lstm: bool) -> Tuple[float, float, float, float]:
    model.eval()
    head.eval()
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels).contiguous().view(-1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if is_lstm:
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)
            outputs2 = head(pooled_output)
            XtoY_output = outputs2[0:1]
            predictions = torch.argmax(XtoY_output[0], axis=1)
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            XtoC_output = outputs2[1:]
            XtoC_logits = torch.cat(XtoC_output, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
    test_acc = float((predict_labels == true_labels).sum()) / float(len(true_labels)) if len(true_labels) else 0.0
    if len(true_labels):
        labels_unique = np.unique(true_labels)
        macro_f1_scores = []
        for lbl in labels_unique:
            macro_f1_scores.append(f1_score(true_labels == lbl, predict_labels == lbl, average='macro'))
        test_macro_f1 = float(np.mean(macro_f1_scores))
    else:
        test_macro_f1 = 0.0
    concept_acc = float((concept_predict_labels == concept_true_labels).sum()) / float(len(concept_true_labels)) if len(concept_true_labels) else 0.0
    if len(concept_true_labels):
        concept_unique = np.unique(concept_true_labels)
        concept_macro_f1_scores = []
        for c in concept_unique:
            concept_macro_f1_scores.append(f1_score(concept_true_labels == c, concept_predict_labels == c, average='macro'))
        concept_macro_f1 = float(np.mean(concept_macro_f1_scores))
    else:
        concept_macro_f1 = 0.0
    return test_acc, test_macro_f1, concept_acc, concept_macro_f1
