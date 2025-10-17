# models/peft_lora.py
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Any
import torch
from config import MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, MODELS_DIR

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

class PEFTLoRAModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Use the best available device
        device = get_best_device()
        self.base_model = self.base_model.to(device)
        print(f"📱 Model loaded on: {device}")
        
    def setup_lora_config(self):
        """Setup LoRA configuration."""
        return LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=LORA_CONFIG["task_type"],
            modules_to_save=LORA_CONFIG["modules_to_save"],
        )
    
    def setup_training_args(self, output_dir: str = None):
        """Setup training arguments with automatic device optimization."""
        output_dir = output_dir or str(MODELS_DIR / "peft-lora-model")
        
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
            print("🚀 CUDA detected: Enabling fp16 mixed precision for LoRA")
        elif device == 'mps':
            # MPS doesn't support mixed precision yet
            fp16 = False
            bf16 = False
            pin_memory = False  # MPS doesn't support pin_memory
            print("🚀 MPS detected: Using full precision for LoRA (mixed precision not supported)")
        else:
            # CPU
            fp16 = False
            bf16 = False
            pin_memory = False
            print("⚠️ CPU detected: Using full precision for LoRA")
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            num_train_epochs=MODEL_CONFIG["num_epochs"],
            learning_rate=2e-4,  # Higher learning rate for LoRA
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
            warmup_steps=TRAINING_CONFIG["warmup_steps"]
        )
    
    def print_trainable_parameters(self):
        """Print trainable parameters."""
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.lora_model.parameters())
        print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}")
    
    def train(self, train_dataset, eval_dataset):
        """Train the model."""
        if self.base_model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
            
        lora_config = self.setup_lora_config()
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        print("LoRA Model Parameter Count:")
        self.print_trainable_parameters()
        
        args = self.setup_training_args()
        
        self.trainer = Trainer(
            model=self.lora_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("Starting PEFT LoRA training...")
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
            
        save_path = save_path or str(MODELS_DIR / "peft-lora-model")
        self.trainer.save_model(save_path)
        print(f"PEFT LoRA model saved to {save_path}")