# CBM (Concept Bottleneck Models) Framework

This directory contains the core implementation of the Concept Bottleneck Models framework for interpreting pretrained language models via high-level, meaningful concepts. The framework supports multiple datasets (CEBaB, IMDB, Essay) and various model backbones (BERT, RoBERTa, GPT-2, LSTM).

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Learning Rate Configuration](#learning-rate-configuration)
- [Training Configuration](#training-configuration)
- [Training Process](#training-process)
- [Output Locations](#output-locations)
- [Usage Examples](#usage-examples)
- [Pipeline Types](#pipeline-types)
- [Model Support](#model-support)
- [Evaluation Metrics](#evaluation-metrics)

## Overview

The CBM framework implements three main pipeline types:

1. **Standard Pipeline** (`get_cbm_standard`): Traditional fine-tuning without concept supervision
2. **Joint Pipeline** (`get_cbm_joint`): Joint training with concept and task supervision
3. **LLM Mix Joint Pipeline** (`get_cbm_LLM_mix_joint`): Joint training with mixed human and machine-generated concepts

## Data Preprocessing

### Supported Datasets

#### CEBaB Dataset
- **Location**: `dataset/cebab/`
- **Variants**:
  - `pure`: Original CEBaB dataset (4 concepts: food, ambiance, service, noise)
  - `aug`: Augmented with additional concepts (10 concepts total)
  - `aug_yelp`: Yelp-exclusive augmented data
  - `aug_both`: Combined CEBaB and Yelp augmented data
- **Concepts**: Food, Ambiance, Service, Noise (base) + Cleanliness, Price, Location, Menu Variety, Waiting Time, Waiting Area (extended)
- **Labels**: 5-star rating system (1-5)

#### IMDB Dataset
- **Location**: `dataset/imdb/`
- **Variants**:
  - `manual`: Human-annotated concepts
  - `gen`: Machine-generated concepts
- **Concepts**: 8 sentiment-related concepts
- **Labels**: Binary sentiment (0-1)

#### Essay Dataset
- **Location**: `dataset/essay/cleaned/`
- **Variants**:
  - `manual`: Human-annotated concepts
  - `generated`: Machine-generated concepts
- **Concepts**: 8 writing quality concepts (FC, CC, TU, CP, R, DU, EE, FR)
- **Labels**: 6-class scoring system (0-5)

### Data Loading

The framework automatically handles data loading through dataset classes:

```python
# CEBaB
train_ds = CEBaBDataset("train", tokenizer, max_len, variant="pure")
val_ds = CEBaBDataset("val", tokenizer, max_len, variant="pure")

# IMDB
train_ds = IMDBDataset("train", tokenizer, max_len, variant="manual")

# Essay
train_ds = EssayDataset("train", tokenizer, max_len, variant="manual")
```

## Learning Rate Configuration

### Finding Optimal Learning Rates

Use the learning rate finder to determine optimal learning rates for your specific dataset and model combination:

```bash
python get_learning_rate.py
```

This script tests multiple learning rate candidates across different models and pipelines, saving results to `lr_finder_results.csv`.

### Pre-configured Learning Rates

The framework includes optimized learning rates for the essay dataset:

```python
# Essay dataset optimized learning rates
LEARNING_RATES = {
    'lstm': 5e-4,           # Joint pipeline optimal
    'gpt2': 5e-5,           # Joint pipeline optimal  
    'roberta-base': 2e-5,   # Joint pipeline optimal
    'bert-base-uncased': 2e-5,  # Joint pipeline optimal
}
```

### Learning Rate Types

- **Dataset Optimal**: Learning rates optimized for specific datasets
- **Universal**: General-purpose learning rates for cross-dataset compatibility

## Training Configuration

### Configuration Parameters

The `RunConfig` class in `config/defaults.py` provides centralized configuration:

```python
@dataclass
class RunConfig:
    mode: str = "standard"                    # Pipeline mode
    model_name: str = "bert-base-uncased"     # Model backbone
    max_len: int = 128                        # Maximum sequence length
    batch_size: int = 8                       # Batch size
    num_epochs: int = 1                       # Number of training epochs
    optimizer_lr: float = 1e-5                # Learning rate
    dataset: str = "cebab"                    # Dataset name
    early_stopping: bool = True               # Enable early stopping
    variant: str = "pure"                     # Dataset variant
```

### Key Configuration Options

- **Early Stopping**: Enabled by default with patience=5 epochs
- **Batch Size**: Default 8 (adjustable based on GPU memory)
- **Max Length**: 128 for CEBaB/IMDB, 512 for Essay
- **Epochs**: 20 for full training, 3 for learning rate finding

## Training Process

### Training Loops

The framework provides specialized training loops for different pipeline types:

#### Standard Training (`training/loops.py`)
- Single-task training without concept supervision
- Standard cross-entropy loss
- Task accuracy and macro-F1 evaluation

#### Joint Training (`training/loops_joint.py`)
- Multi-task training with concept and task supervision
- Combined loss: `λ * concept_loss + task_loss`
- Separate evaluation for concept and task performance

### Training Features

- **Early Stopping**: Prevents overfitting with configurable patience
- **Model Checkpointing**: Saves best models during training
- **Learning Rate Scheduling**: StepLR for LSTM models
- **Mixed Precision**: Automatic GPU optimization

### Training Command

```bash
# Run essay experiments with optimal learning rates
python run_essay.py

# Enable early stopping
python run_essay.py --early-stopping

# Run comprehensive experiments
python main.py
```

## Output Locations

### Model Checkpoints

Models are saved in two formats:

#### Original Format (`.pth`)
- **Location**: `saved_models/original/<model_name>/`
- **Files**: Complete model objects with full state
- **Usage**: Direct loading with `torch.load()`

#### Portable Format (`.pt`)
- **Location**: `saved_models/portable/<model_name>/`
- **Files**: State dictionaries only
- **Usage**: Load with `model.load_state_dict()`

### Result Files

#### CSV Results
- **Main Results**: `result_essay.csv`, `result_test.csv`
- **Test Results**: `cbm/tests/test_results/`
- **Learning Rate Results**: `lr_finder_results.csv`

#### Result Format
```csv
dataset,data_type,function,model,score,concept_score
essay,D,PLMs,bert-base-uncased,"[(0.85, 0.82)]","[]"
essay,D,CBE-PLMs,bert-base-uncased,"[(0.87, 0.84)]","[(0.78, 0.75)]"
```

### Logs and Analysis
- **Training Logs**: Console output with epoch-by-epoch metrics
- **Analysis Files**: `cbm/tests/test_results/*.md`
- **Pivot Tables**: Formatted results for paper tables

## Usage Examples

### Basic Training

```python
from cbm import get_cbm_standard, get_cbm_joint

# Standard pipeline (no concepts)
scores = get_cbm_standard(
    model_name="bert-base-uncased",
    dataset="essay",
    variant="manual",
    num_epochs=20,
    optimizer_lr=2e-5
)

# Joint pipeline (with concepts)
results = get_cbm_joint(
    model_name="bert-base-uncased", 
    dataset="essay",
    variant="manual",
    num_epochs=20,
    optimizer_lr=2e-5
)
```

### Custom Configuration

```python
from cbm.config.defaults import make_run_config

config = make_run_config(
    model_name="roberta-base",
    dataset="cebab",
    variant="aug",
    max_len=256,
    batch_size=16,
    num_epochs=30,
    optimizer_lr=1e-5,
    early_stopping=True
)
```

### Learning Rate Finding

```python
from get_learning_rate import quick_lr_test

# Test learning rates for a specific model
results = quick_lr_test(
    model_name="bert-base-uncased",
    pipeline="joint",
    variant="manual",
    num_epochs=3
)
```

## Pipeline Types

### 1. Standard Pipeline (`get_cbm_standard`)
- **Purpose**: Baseline fine-tuning without concept supervision
- **Architecture**: `Input → Encoder → Classifier → Task Labels`
- **Loss**: Cross-entropy on task labels only
- **Use Case**: Performance comparison baseline

### 2. Joint Pipeline (`get_cbm_joint`)
- **Purpose**: Joint training with concept and task supervision
- **Architecture**: `Input → Encoder → Concept Classifier → Task Classifier`
- **Loss**: `λ * concept_loss + task_loss`
- **Use Case**: Concept-aware training with human annotations

### 3. LLM Mix Joint Pipeline (`get_cbm_LLM_mix_joint`)
- **Purpose**: Joint training with mixed human and machine concepts
- **Architecture**: Similar to joint but with concept mixing
- **Loss**: Combined loss with concept mixing weights
- **Use Case**: Leveraging both human and machine-generated concepts

## Model Support

### Supported Backbones

| Model | Tokenizer | Hidden Size | Special Notes |
|-------|-----------|-------------|---------------|
| BERT-base-uncased | BertTokenizer | 768 | Standard transformer |
| RoBERTa-base | RobertaTokenizer | 768 | Improved BERT variant |
| GPT-2 | GPT2Tokenizer | 768 | Decoder-only model |
| LSTM | BertTokenizer | 128 | BiLSTM with attention |

### Model Loading

```python
from cbm.models.loaders import load_model_and_tokenizer

model, tokenizer, hidden_size = load_model_and_tokenizer(
    model_name="bert-base-uncased",
    fasttext_path=None  # Optional for LSTM
)
```

## Evaluation Metrics

### Task Metrics
- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Average F1 score across all classes

### Concept Metrics
- **Concept Accuracy**: Accuracy on concept prediction
- **Concept Macro-F1**: Average F1 score for concept classification

### Evaluation Process

```python
from cbm.evaluation.metrics import compute_accuracy, compute_macro_f1

# Compute metrics
acc = compute_accuracy(predictions, labels)
f1 = compute_macro_f1(predictions, labels)
```

## File Structure

```
cbm/
├── config/
│   └── defaults.py          # Configuration management
├── data/
│   ├── cebab.py            # CEBaB dataset loader
│   ├── imdb.py             # IMDB dataset loader
│   └── essay.py            # Essay dataset loader
├── evaluation/
│   └── metrics.py          # Evaluation metrics
├── models/
│   └── loaders.py          # Model and tokenizer loading
├── pipelines/
│   ├── standard.py         # Standard pipeline
│   ├── joint.py            # Joint pipeline
│   └── llm_mix_joint.py    # LLM mix joint pipeline
├── training/
│   ├── loops.py            # Standard training loops
│   ├── loops_joint.py      # Joint training loops
│   └── mixup.py            # Data augmentation
├── tests/
│   └── test_results/       # Test results and analysis
├── main.py                 # Main experiment runner
└── run_essay.py           # Essay-specific experiments
```

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0
- tqdm >= 4.60.0
- datasets (for CEBaB)

