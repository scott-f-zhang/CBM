import os
import pandas as pd
from typing import Dict, Optional

def load_learning_rates(dataset: str) -> Dict[str, float]:
    """从 CSV 文件加载学习率配置
    
    Args:
        dataset: 数据集名称 (essay, qa, cebab, imdb)
    
    Returns:
        Dict mapping model_name -> learning_rate
        
    Raises:
        FileNotFoundError: 如果找不到对应的 CSV 文件
    """
    # 定位 cbm/lr_rate/<dataset>_lr_rate.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cbm_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(cbm_dir, "lr_rate", f"{dataset}_lr_rate.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Learning rate file not found: {csv_path}")
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 转换为字典
    lr_dict = dict(zip(df['model'], df['best_lr']))
    
    return lr_dict
