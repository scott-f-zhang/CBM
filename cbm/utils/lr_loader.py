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
        FileNotFoundError: If the corresponding CSV file is not found
    """
    # Locate cbm/lr_rate/<dataset>_lr_rate.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cbm_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(cbm_dir, "lr_rate", f"{dataset}_lr_rate.csv")
    
    # Add log: show where to load from
    print(f"📁 Loading learning rate configuration from file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Learning rate file not found: {csv_path}")
        raise FileNotFoundError(f"Learning rate file not found: {csv_path}")
    
    # Read CSV
    print(f"📖 Reading learning rate file...")
    df = pd.read_csv(csv_path)
    
    # Convert to dictionary
    lr_dict = dict(zip(df['model'], df['best_lr']))
    
    # Add log: show loaded content
    print(f"✅ Learning rate configuration loaded successfully, containing {len(lr_dict)} model configurations:")
    for model, lr in lr_dict.items():
        print(f"   - {model}: {lr}")
    
    return lr_dict
