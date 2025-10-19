# Unified Hyperparameters for Fair Comparison

## 🎯 Objective
To demonstrate that **CBE-PLMs outperform PLMs**, all hyperparameters must be **identical** between the two approaches to ensure fair comparison.

## 📊 Unified Hyperparameter Settings

### **Training Configuration**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Epochs** | 20 | Aligned with run_cebab and main/main.py |
| **Batch Size** | 8 | Consistent across all experiments |
| **Max Length** | 512 | Standard for transformer models |
| **Early Stopping** | Patience=5 | Prevents overfitting |

### **Learning Rates (Dataset-Optimized)**
| Model | Learning Rate | Source |
|-------|---------------|--------|
| **BERT** | 2e-5 | Essay dataset LR finder (Joint optimal) |
| **RoBERTa** | 2e-5 | Essay dataset LR finder (Joint optimal) |
| **GPT2** | 5e-5 | Essay dataset LR finder (Joint optimal) |
| **LSTM** | 5e-4 | Essay dataset LR finder (Joint optimal) |

### **Model Architecture**
| Component | Setting | Justification |
|-----------|---------|---------------|
| **Backbone** | Same for PLMs/CBE-PLMs | Fair comparison |
| **Hidden Size** | Model-specific | Preserve model characteristics |
| **Concept Head** | 8 concepts × 3 classes | Essay dataset specific |
| **Task Head** | 2 classes (binary) | Essay dataset specific |

## 🔬 Scientific Rigor

### **Controlled Variables**
✅ **Learning Rate**: Identical for both approaches  
✅ **Training Epochs**: Identical (20 epochs)  
✅ **Batch Size**: Identical (8)  
✅ **Max Sequence Length**: Identical (512)  
✅ **Early Stopping**: Identical (patience=5)  
✅ **Data Split**: Identical (7:2:1 train:dev:test)  
✅ **Random Seeds**: Should be fixed for reproducibility  

### **Independent Variable**
🎯 **Architecture**: PLMs vs CBE-PLMs (only difference)

### **Dependent Variables**
📊 **Task Accuracy**: Primary metric  
📊 **Task F1**: Secondary metric  
📊 **Concept Accuracy**: CBE-PLMs specific  
📊 **Concept F1**: CBE-PLMs specific  

## 📈 Expected Results

With unified hyperparameters, we expect:

1. **Fair Comparison**: Any performance difference is due to architecture, not hyperparameters
2. **Reproducible Results**: Same settings across all experiments
3. **Scientific Validity**: Controlled experiment design
4. **Clear Interpretation**: Results directly attributable to CBE-PLMs vs PLMs

## 🚀 Implementation

```python
# Unified settings for all experiments
BASE_RUN = RunConfig(
    num_epochs=20,      # Unified with run_cebab/main
    max_len=512,        # Standard setting
    batch_size=8,       # Consistent across experiments
)

# Dataset-optimized learning rates (from LR finder)
LEARNING_RATES = {
    'bert-base-uncased': 2e-5,  # Joint optimal from LR finder
    'roberta-base': 2e-5,       # Joint optimal from LR finder
    'gpt2': 5e-5,               # Joint optimal from LR finder
    'lstm': 5e-4,               # Joint optimal from LR finder
}
```

## 📝 Notes

- **No hyperparameter tuning** for individual approaches
- **Same computational budget** for both PLMs and CBE-PLMs
- **Identical training conditions** ensure fair comparison
- **Results are directly comparable** and interpretable

This unified approach ensures that any performance difference between PLMs and CBE-PLMs is due to the architectural difference (concept bottleneck), not due to different hyperparameter settings.
