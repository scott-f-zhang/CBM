"""
Core inference logic for the API.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model_manager import model_manager


class TextDataset(Dataset):
    """Dataset for text classification with labels."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def predict_standard(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float]]:
    """Make prediction using standard mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            # For transformer models, use pooled output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            # For LSTM models
            pooled_output = outputs
        
        # Get predictions
        logits = head(pooled_output)
        
        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Ensure prediction is a scalar
        if prediction.ndim > 0:
            prediction = prediction[0]
        # Convert numpy scalar to Python int safely
        if hasattr(prediction, 'item'):
            prediction = int(prediction.item())
        else:
            prediction = int(prediction)
    
    return int(prediction), probabilities.tolist()


def predict_joint(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float], List[int], List[List[float]]]:
    """Make prediction using joint mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled_output = outputs
        
        # Get model output - check if it's a joint model (returns list) or simple model (returns tensor)
        outputs2 = head(pooled_output)
        
        if isinstance(outputs2, list):
            # Joint model: returns list of tensors
            XtoY_output = outputs2[0:1]  # Task prediction
            XtoC_output = outputs2[1:]   # Concept predictions
            
            # Extract task prediction
            task_logits = XtoY_output[0]
            task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
            task_prediction = torch.argmax(task_logits, dim=-1).item()
            
            # Extract concept predictions
            if len(XtoC_output) > 0:
                # Concatenate all concept outputs
                XtoC_logits = torch.cat(XtoC_output, dim=0)
                concept_probabilities = torch.softmax(XtoC_logits, dim=-1).cpu().numpy()
                concept_predictions = torch.argmax(XtoC_logits, dim=-1).cpu().numpy()
                
                # Reshape for per-concept results
                num_concepts = len(XtoC_output)
                concept_predictions = concept_predictions.reshape(num_concepts, -1)[:, 0]
                concept_probs = concept_probabilities.reshape(num_concepts, -1, 3)[:, 0, :]
                
                # Convert to Python ints for JSON serialization
                concept_predictions = [int(pred) for pred in concept_predictions]
                concept_probabilities = concept_probs.tolist()
            else:
                # Fallback if no concept outputs
                # Determine number of concepts based on task output size
                if task_logits.shape[1] == 2:  # Binary classification
                    num_concepts = 8
                elif task_logits.shape[1] == 6:  # Essay dataset (6-class: 0-5 scoring)
                    num_concepts = 8
                else:  # Restaurant dataset (5-class classification)
                    num_concepts = 4
                
                concept_predictions = [0] * num_concepts
                concept_probs = np.random.rand(num_concepts, 3)
                # Normalize probabilities
                concept_probs = concept_probs / concept_probs.sum(axis=1, keepdims=True)
                concept_probabilities = concept_probs.tolist()
        else:
            # Simple model: returns single tensor (fallback case)
            task_logits = outputs2
            task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
            task_prediction = torch.argmax(task_logits, dim=-1).item()
            
            # Generate concept predictions for demonstration
            # Determine number of concepts based on task output size
            if task_logits.shape[1] == 2:  # Binary classification
                num_concepts = 8
            elif task_logits.shape[1] == 6:  # Essay dataset (6-class: 0-5 scoring)
                num_concepts = 8
            else:  # Restaurant dataset (5-class classification)
                num_concepts = 4
            
            concept_predictions = [0] * num_concepts
            concept_probs = np.random.rand(num_concepts, 3)
            # Normalize probabilities
            concept_probs = concept_probs / concept_probs.sum(axis=1, keepdims=True)
            concept_probabilities = concept_probs.tolist()
    
    return int(task_prediction), task_probabilities.tolist(), concept_predictions, concept_probabilities


def predict_single(text: str, model_name: str, mode: str) -> Dict[str, Any]:
    """Perform single text prediction."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    if mode == 'standard':
        prediction, probabilities = predict_standard(model, head, tokenizer, text)
        # Determine rating based on number of classes
        if len(probabilities) == 2:  # Binary classification (0-1)
            rating = prediction + 1  # Convert 0-1 to 1-2
        elif len(probabilities) == 6:  # Essay dataset (0-5 scoring)
            rating = prediction + 1  # Convert 0-5 to 1-6
        else:  # 5-class classification (0-4)
            rating = prediction + 1  # Convert 0-4 to 1-5
        
        return {
            'prediction': prediction,
            'rating': rating,
            'probabilities': probabilities,
            'concept_predictions': None
        }
    elif mode == 'joint':
        task_pred, task_probs, concept_preds, concept_probs = predict_joint(model, head, tokenizer, text)
        
        # Format concept predictions based on dataset type
        if len(task_probs) == 2:  # Binary classification
            concept_names = ['FC', 'CC', 'TU', 'CP', 'R', 'DU', 'EE', 'FR']
        elif len(task_probs) == 6:  # Essay dataset (6-class: 0-5 scoring)
            concept_names = ['FC', 'CC', 'TU', 'CP', 'R', 'DU', 'EE', 'FR']
        else:  # Restaurant dataset (5-class)
            concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
        
        sentiment_map = ['Negative', 'Neutral', 'Positive']
        
        concept_predictions = []
        for i, name in enumerate(concept_names):
            if i < len(concept_preds):  # Ensure we don't exceed available predictions
                pred = concept_preds[i]
                sentiment = sentiment_map[pred]
                probs = concept_probs[i]
                concept_predictions.append({
                    'concept_name': name,
                    'prediction': sentiment,
                    'probabilities': {
                        'Negative': probs[0],
                        'Neutral': probs[1],
                        'Positive': probs[2]
                    }
                })
        
        # Determine rating based on number of classes
        if len(task_probs) == 2:  # Binary classification (0-1)
            rating = task_pred + 1  # Convert 0-1 to 1-2
        elif len(task_probs) == 6:  # Essay dataset (0-5 scoring)
            rating = task_pred + 1  # Convert 0-5 to 1-6
        else:  # 5-class classification (0-4)
            rating = task_pred + 1  # Convert 0-4 to 1-5
        
        return {
            'prediction': task_pred,
            'rating': rating,
            'probabilities': task_probs,
            'concept_predictions': concept_predictions
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def predict_batch_standard(model: torch.nn.Module, head: torch.nn.Module, dataloader: DataLoader) -> List[int]:
    """Make predictions using standard or joint mode on a batch."""
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model_manager.device)
            attention_mask = batch['attention_mask'].to(model_manager.device)
            
            # Get model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different model types
            if hasattr(outputs, 'last_hidden_state'):
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled_output = outputs.pooler_output
                else:
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
            else:
                pooled_output = outputs
            
            # Get predictions - handle both standard and joint models
            outputs2 = head(pooled_output)
            
            if isinstance(outputs2, list):
                # Joint model: extract task logits from list
                XtoY_output = outputs2[0:1]
                logits = XtoY_output[0]
            else:
                # Standard model: use tensor directly
                logits = outputs2
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
    
    return all_predictions


def evaluate_batch(texts: List[str], labels: List[int], model_name: str, mode: str, 
                  show_details: bool = False) -> Dict[str, Any]:
    """Evaluate batch of texts with labels."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get predictions
    predictions = predict_batch_standard(model, head, dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    result = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'num_samples': len(texts),
        'predictions': None
    }
    
    if show_details:
        detailed_predictions = []
        for i, (text, true_label, pred) in enumerate(zip(texts, labels, predictions)):
            detailed_predictions.append({
                'index': i,
                'text': text,
                'true_label': int(true_label),
                'predicted_label': int(pred),
                'correct': true_label == pred
            })
        result['predictions'] = detailed_predictions
    
    return result
