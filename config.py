import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "results" / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Model configuration
MODEL_CONFIG = {
    "model_name": "dmis-lab/biobert-base-cased-v1.1",
    "max_length": 384,
    "stride": 128,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "weight_decay": 0.01,
}

# LoRA configuration
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "key", "value"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "QUESTION_ANS",
    "modules_to_save": ["qa_outputs"],
}

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "logging_strategy": "steps",
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 500,
    "load_best_model_at_end": True,
    "fp16": False,  # Will be auto-detected based on device
    "bf16": False,  # Will be auto-detected based on device
    "dataloader_pin_memory": True,  # Enable for better performance on supported devices
    "warmup_steps": 200,
}

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)