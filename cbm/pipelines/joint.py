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
from ..training.loops_joint import train_epoch_joint, eval_epoch_joint, test_epoch_joint

from ..models.cbm_models import ModelXtoCtoY_function


def get_cbm_joint(
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
        default_dataset='cebab', default_variant='aug',
    )
    cfg.mode = 'joint' if cfg.mode is None else cfg.mode

    lambda_XtoC = 0.5
    is_aux_logits = False
    num_labels = 5
    num_each_concept_classes = 3

    model, tokenizer, hidden_size = load_model_and_tokenizer(cfg.model_name, fasttext_path=fasttext_path)

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
        num_each_concept_classes = len(train_ds.concept_vals)
    elif cfg.dataset == 'qa':
        train_ds = QADataset("train", tokenizer, cfg.max_len)
        val_ds = QADataset("val", tokenizer, cfg.max_len)
        test_ds = QADataset("test", tokenizer, cfg.max_len)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)
    else:
        train_ds = CEBaBDataset("train", tokenizer, cfg.max_len, variant=cfg.variant)
        val_ds = CEBaBDataset("val", tokenizer, cfg.max_len, variant=cfg.variant)
        test_ds = CEBaBDataset("test", tokenizer, cfg.max_len, variant=cfg.variant)
        num_labels = len(train_ds.final_label_vals)
        num_concept_labels = len(train_ds.concepts)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # num_concept_labels already set above per dataset

    if cfg.model_name == 'lstm':
        head = ModelXtoCtoY_function(
            concept_classes=num_each_concept_classes, label_classes=num_labels, n_attributes=num_concept_labels,
            bottleneck=True, expand_dim=0, n_class_attr=num_each_concept_classes,
            use_relu=False, use_sigmoid=False, Lstm=True, aux_logits=is_aux_logits,
        )
    else:
        head = ModelXtoCtoY_function(
            concept_classes=num_each_concept_classes, label_classes=num_labels, n_attributes=num_concept_labels,
            bottleneck=True, expand_dim=0, n_class_attr=num_each_concept_classes,
            use_relu=False, use_sigmoid=False, aux_logits=is_aux_logits,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    # Prepare save directory: <project_root>/saved_models/<dataset>/<model_name>/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(PROJECT_ROOT, "saved_models", cfg.dataset, cfg.model_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{cfg.model_name}_joint.pth")
    head_path = os.path.join(save_dir, f"{cfg.model_name}_ModelXtoCtoY_layer_joint.pth")

    default_lr = 1e-2 if cfg.model_name == 'lstm' else 1e-5
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=(cfg.optimizer_lr if cfg.optimizer_lr is not None else default_lr))
    if cfg.model_name == 'lstm':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = (cfg.early_stopping_patience if cfg.early_stopping else float('inf'))  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(cfg.num_epochs):
        train_epoch_joint(
            model,
            head,
            train_loader,
            device,
            {"optimizer": optimizer, "criterion": criterion},
            lambda_XtoC,
            cfg.model_name == 'lstm',
        )
        metrics = eval_epoch_joint(model, head, val_loader, device, cfg.model_name == 'lstm')
        print(
            f"Epoch {epoch + 1}: Val concept Acc = {metrics['concept_acc']*100} "
            f"Val concept Macro F1 = {metrics['concept_macro_f1']*100}"
        )
        print(f"Epoch {epoch + 1}: Val Acc = {metrics['val_acc']*100} Val Macro F1 = {metrics['val_macro_f1']*100}")
        
        # Early stopping logic
        if metrics['val_acc'] > best_acc:
            best_acc = metrics['val_acc']
            patience_counter = 0  # Reset patience counter
            # Save full pickled objects (.pth)
            torch.save(model, model_path)
            torch.save(head, head_path)
            print(f"  -> New best validation accuracy: {best_acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epochs (patience: {patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

        # Step LR scheduler for LSTM after each epoch
        if cfg.model_name == 'lstm':
            scheduler.step()

    model = torch.load(model_path, weights_only=False)
    head = torch.load(head_path, weights_only=False)
    model.to(device)
    head.to(device)
    
    test_acc, test_macro_f1, concept_acc, concept_macro_f1 = test_epoch_joint(model, head, test_loader, device, cfg.model_name == 'lstm')
    print(f"Epoch 1: Test Acc = {test_acc*100} Test Macro F1 = {test_macro_f1*100}")
    return {
        'task': [(test_acc, test_macro_f1)],
        'concept': [(concept_acc, concept_macro_f1)],
    }
