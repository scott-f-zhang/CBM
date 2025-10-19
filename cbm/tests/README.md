# CBM-NLP Tests

This directory contains test scripts for the CBM-NLP project.

## Test Files

### Dataset-Specific Tests (Recommended)
- **`test_essay.py`**: Essay dataset experiments with configurable early stopping
- **`test_cebab.py`**: CEBaB dataset test runner with flexible model selection
- **`test_imdb.py`**: IMDB dataset test runner with flexible model selection

### Data Tests
- **`test_dataclass_loading.py`**: Data loading functionality tests

## Running Tests

### From Project Root
```bash
# Run complete essay experiments (recommended - NO early stopping by default)
python main/run_essay.py

# Run with early stopping enabled
python main/run_essay.py --early-stopping

# Run with early stopping explicitly disabled
python main/run_essay.py --no-early-stopping

# Run essay experiments (with early stopping by default)
python main/tests/test_essay.py

# Run essay experiments without early stopping
python main/tests/test_essay.py --no_early_stopping

# Run CEBaB dataset tests (all models)
python main/tests/test_cebab.py --model all

# Run specific model tests on CEBaB
python main/tests/test_cebab.py --model bert
python main/tests/test_cebab.py --model gpt2 --variants D^

# Run IMDB dataset tests (all models)
python main/tests/test_imdb.py --model all

# Run specific model tests on IMDB
python main/tests/test_imdb.py --model bert --variants D^
```

### From Main Directory
```bash
cd main

# Run complete essay experiments (recommended - NO early stopping by default)
python run_essay.py

# Run with early stopping enabled
python run_essay.py --early-stopping

# Run essay experiments (with early stopping by default)
python tests/test_essay.py

# Run essay experiments without early stopping
python tests/test_essay.py --no_early_stopping

# Run CEBaB dataset tests
python tests/test_cebab.py --model all

# Run IMDB dataset tests
python tests/test_imdb.py --model all
```

### From Tests Directory
```bash
cd main/tests

# Run essay experiments (with early stopping by default)
python test_essay.py

# Run essay experiments without early stopping
python test_essay.py --no_early_stopping

# Run CEBaB dataset tests
python test_cebab.py --model all

# Run IMDB dataset tests
python test_imdb.py --model all
```

## Environment Setup

Make sure to activate the conda environment before running tests:
```bash
conda activate cbm
```

## Test Results

All test results are automatically saved to the `test_results/` directory:

```
main/tests/test_results/
├── result_essay_early_stopping_dataset_optimal.csv      # Essay experiments WITH early stopping
├── result_essay_no_early_stopping_dataset_optimal.csv   # Essay experiments WITHOUT early stopping
├── result_cebab_all_early_stopping.csv                  # CEBaB all models WITH early stopping
├── result_cebab_bert_no_early_stopping.csv              # CEBaB BERT WITHOUT early stopping
├── result_imdb_all_early_stopping.csv                   # IMDB all models WITH early stopping
├── result_imdb_gpt2_no_early_stopping.csv               # IMDB GPT2 WITHOUT early stopping
├── lr_finder_results.csv                                # Learning rate finder results
├── early_stopping_analysis.md                           # Analysis of early stopping impact
└── lr_analysis_summary.md                               # Learning rate analysis summary
```

## Notes

- **Dataset-specific test files**: Tests are organized by dataset for better clarity
- **Model selection**: All dataset tests support `--model` parameter for flexible model selection
- **Early stopping control**: Available via `--early_stopping`/`--no_early_stopping` flags
- **Default behavior**: Early stopping ENABLED by default in all tests
- **Configuration**: Early stopping parameter added to `RunConfig` in `main/config/defaults.py`
- **Independence**: `run_essay.py` is now independent of test files and contains its own core logic
- All results are automatically saved to `main/tests/test_results/` directory
- Early Stopping is implemented in both PLMs and CBE-PLMs pipelines
- Tests use appropriate train/dev/test splits for each dataset
- **Recommended**: Use `run_essay.py` for main experiments, test files for specific testing scenarios
