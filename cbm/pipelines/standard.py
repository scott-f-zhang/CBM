from typing import Optional
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from ..config.defaults import make_run_config
from ..models.loaders import load_model_and_tokenizer
from ..data.cebab import CEBaBDataset
from ..data.imdb import IMDBDataset
from ..data.qa import QADataset
from ..data.essay import EssayDataset
from ..training.loops import train_one_epoch, evaluate, test_loop


def get_cbm_standard(
    mode=None,
    max_len=None,
    batch_size=None,
    model_name=None,
    num_epochs=None,
    optimizer_lr=None,
    dataset: Optional[str] = None,
    variant: Optional[str] = None,
    early_stopping: Optional[bool] = None,
    fasttext_path: Optional[str] = None,
):
    cfg = make_run_config(
        mode=mode, max_len=max_len, batch_size=batch_size, model_name=model_name,
        num_epochs=num_epochs, optimizer_lr=optimizer_lr,
        dataset=dataset, variant=variant, early_stopping=early_stopping,
        default_dataset='cebab', default_variant='pure',
    )
    cfg.mode = 'standard' if cfg.mode is None else cfg.mode

    model, tokenizer, hidden_size = load_model_and_tokenizer(cfg.model_name, fasttext_path=fasttext_path)

    # Dataset selection: cebab / imdb / essay / qa
    if cfg.dataset == 'imdb':
        train_ds = IMDBDataset("train", tokenizer, cfg.max_len, variant=cfg.variant)
        val_ds = IMDBDataset("val", tokenizer, cfg.max_len, variant=cfg.variant)
        test_ds = IMDBDataset("test", tokenizer, cfg.max_len, variant=cfg.variant)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)
    elif cfg.dataset == 'essay':
        train_ds = EssayDataset("train", tokenizer, cfg.max_len)
        val_ds = EssayDataset("val", tokenizer, cfg.max_len)
        test_ds = EssayDataset("test", tokenizer, cfg.max_len)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)
    elif cfg.dataset == 'qa':
        train_ds = QADataset("train", tokenizer, cfg.max_len)
        val_ds = QADataset("val", tokenizer, cfg.max_len)
        test_ds = QADataset("test", tokenizer, cfg.max_len)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)
    else:
        # cebab
        train_ds = CEBaBDataset("train", tokenizer, cfg.max_len, variant=cfg.variant, expand_concepts=None)
        val_ds = CEBaBDataset("val", tokenizer, cfg.max_len, variant=cfg.variant, expand_concepts=None)
        test_ds = CEBaBDataset("test", tokenizer, cfg.max_len, variant=cfg.variant, expand_concepts=None)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # num_labels and num_concept_labels set above per dataset
    if cfg.model_name == 'lstm':
        head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Linear(hidden_size // 2, num_concept_labels),
            torch.nn.Linear(num_concept_labels, num_labels),
        )
        scheduler_needed = True
    else:
        head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_concept_labels),
            torch.nn.Linear(num_concept_labels, num_labels),
        )
        scheduler_needed = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    # Prepare save directory: <project_root>/saved_models/<dataset>/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(PROJECT_ROOT, "saved_models", cfg.dataset)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{cfg.model_name}_model_standard.pth")
    head_path = os.path.join(save_dir, f"{cfg.model_name}_classifier_standard.pth")

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=(cfg.optimizer_lr if cfg.optimizer_lr is not None else (1e-2 if cfg.model_name == 'lstm' else 1e-5)))
    if scheduler_needed:
        _scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 5 if cfg.early_stopping else float('inf')  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(cfg.num_epochs):
        train_one_epoch(model, head, train_loader, device, criterion, optimizer, cfg.model_name == 'lstm')
        val_acc, val_f1 = evaluate(model, head, val_loader, device, cfg.model_name == 'lstm')
        print(f"Epoch {epoch + 1}: Val Acc = {val_acc*100} Val Macro F1 = {val_f1*100}")
        
        # Early stopping logic
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # Reset patience counter
            # Save full pickled objects (.pth)
            torch.save(head, head_path)
            torch.save(model, model_path)
            print(f"  -> New best validation accuracy: {best_acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epochs (patience: {patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    # test
    model = torch.load(model_path, weights_only=False)
    head = torch.load(head_path, weights_only=False)
    model.to(device)
    head.to(device)
    
    # Portable state_dict saving has been removed per project requirements.

    scores = test_loop(model, head, test_loader, device, cfg.model_name == 'lstm')
    return scores
