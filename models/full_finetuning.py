# models/full_finetuning.py
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer
)
from typing import Dict, Any, Tuple
import torch
from config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR

def get_best_device():
    """Automatically detect and return the best available device."""
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

class FullFineTuningModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Use the best available device
        device = get_best_device()
        self.model = self.model.to(device)
        print(f"📱 Model loaded on: {device}")
        
    def setup_training_args(self, output_dir: str = None):
        """Setup training arguments with automatic device optimization."""
        output_dir = output_dir or str(MODELS_DIR / "full-finetuned-model")
        
        # Auto-detect optimal settings based on device
        device = get_best_device()
        
        # Configure mixed precision based on device capabilities
        fp16 = False
        bf16 = False
        pin_memory = TRAINING_CONFIG["dataloader_pin_memory"]
        
        if device == 'cuda':
            # CUDA supports both fp16 and bf16
            fp16 = True  # Use fp16 for CUDA
            pin_memory = True
            print("🚀 CUDA detected: Enabling fp16 mixed precision")
        elif device == 'mps':
            # MPS doesn't support mixed precision yet
            fp16 = False
            bf16 = False
            pin_memory = False  # MPS doesn't support pin_memory
            print("🚀 MPS detected: Using full precision (mixed precision not supported)")
        else:
            # CPU
            fp16 = False
            bf16 = False
            pin_memory = False
            print("⚠️ CPU detected: Using full precision")
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            num_train_epochs=MODEL_CONFIG["num_epochs"],
            learning_rate=MODEL_CONFIG["learning_rate"],
            eval_strategy=TRAINING_CONFIG["eval_strategy"],
            eval_steps=TRAINING_CONFIG["eval_steps"],
            logging_strategy=TRAINING_CONFIG["logging_strategy"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_strategy=TRAINING_CONFIG["save_strategy"],
            save_steps=TRAINING_CONFIG["save_steps"],
            load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
            weight_decay=MODEL_CONFIG["weight_decay"],
            fp16=fp16,
            bf16=bf16,
            dataloader_pin_memory=pin_memory,
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
        )
    
    def train(self, train_dataset, eval_dataset):
        """Train the model."""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
            
        args = self.setup_training_args()
        
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("Starting full fine-tuning training...")
        self.trainer.train()
        print("Training completed!")
        
        return self.trainer
    
    def predict(self, test_dataset):
        """Make predictions on test dataset."""
        if self.trainer is None:
            raise ValueError("Model must be trained first")
            
        return self.trainer.predict(test_dataset)
    
    def save_model(self, save_path: str = None):
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError("Model must be trained first")
            
        save_path = save_path or str(MODELS_DIR / "full-finetuned-model")
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")