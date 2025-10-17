# data/data_loader.py
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import Dict, Any, Tuple
import torch

def load_bioasq_data(data_path: str) -> Dict[str, Any]:
    """Load BioASQ data from JSON file."""
    with Path(data_path).open() as json_file:
        data = json.load(json_file)
    return data

def extract_data(json_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract data from BioASQ JSON format into DataFrame."""
    data_rows = []
    
    for passage in json_data['data'][0]['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            qa_id = qa['id']
            
            if qa['answers']:
                answer_text = qa['answers'][0]['text']
                answer_start = qa['answers'][0]['answer_start']
                answer_end = answer_start + len(answer_text)

                data_rows.append({
                    'id': qa_id,
                    'question': question,
                    'answer': answer_text,
                    'context': context,
                    'answer_start': answer_start,
                    'answer_end': answer_end
                })

    return pd.DataFrame(data_rows)

def create_train_val_test_split(df: pd.DataFrame, 
                               train_size: float = 0.8, 
                               val_size: float = 0.1,
                               test_size: float = 0.1,
                               random_state: int = 42) -> DatasetDict:
    """Create train/validation/test splits."""
    full_dataset = Dataset.from_pandas(df)
    
    # First split: train (80%) and temp (20%)
    train_test_split = full_dataset.train_test_split(
        test_size=0.2, seed=random_state
    )
    
    # Second split: validation (10%) and test (10%)
    test_validation_split = train_test_split['test'].train_test_split(
        test_size=0.5, seed=random_state
    )
    
    return DatasetDict({
        'train': train_test_split['train'],
        'validation': test_validation_split['train'],
        'test': test_validation_split['test']
    })

def setup_device() -> str:
    """Setup device for training with automatic detection."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Using Apple Silicon (MPS)")
    else:
        device = 'cpu'
        print("⚠️ Using CPU (no GPU/MPS available)")
    
    return device