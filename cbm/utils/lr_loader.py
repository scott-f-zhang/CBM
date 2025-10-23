import os
import pandas as pd
from typing import Dict, Optional

def load_learning_rates(dataset: str) -> Dict[str, float]:
    """Load learning rate configuration from CSV file
    
    Args:
        dataset: Dataset name (essay, qa, cebab, imdb)
    
    Returns:
        Dict mapping model_name -> learning_rate
        
    Raises:
        FileNotFoundError: If corresponding CSV file is not found
    """
    # Locate cbm/lr_rate/<dataset>_lr_rate.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cbm_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(cbm_dir, "lr_rate", f"{dataset}_lr_rate.csv")
    
    # Log the file path being loaded from
    print(f"Loading learning rates from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Learning rate file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert to dictionary
    lr_dict = dict(zip(df['model'], df['best_lr']))
    
    # Log the loaded learning rates
    print(f"Loaded learning rates: {lr_dict}")
    
    return lr_dict
