# utils/metrics.py
import evaluate
import numpy as np
from typing import Dict, List, Any

def calculate_squad_metrics(predictions: Dict[str, str], 
                           references: List[Dict]) -> Dict[str, float]:
    """Calculate SQuAD metrics (EM and F1)."""
    squad_metric = evaluate.load("squad")
    
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    
    formatted_references = [
        {
            "id": ex["id"], 
            "answers": {
                "text": [ex["answer"]], 
                "answer_start": [ex["answer_start"]]
            }
        } for ex in references
    ]
    
    squad_results = squad_metric.compute(
        predictions=formatted_predictions, 
        references=formatted_references
    )
    
    return {
        "exact_match": np.mean(squad_results["exact_match"]) / 100,
        "f1": np.mean(squad_results["f1"]) / 100
    }

def calculate_bertscore(predictions: Dict[str, str], 
                       references: List[Dict]) -> Dict[str, float]:
    """Calculate BERTScore metrics."""
    bertscore_metric = evaluate.load("bertscore")
    
    predictions_list = list(predictions.values())
    references_list = [ex["answer"] for ex in references]
    
    bertscore_results = bertscore_metric.compute(
        predictions=predictions_list, 
        references=references_list, 
        lang="en"
    )
    
    return {
        "precision": np.mean(bertscore_results["precision"]),
        "recall": np.mean(bertscore_results["recall"]),
        "f1": np.mean(bertscore_results["f1"])
    }

def compare_models(fft_results: Dict, lora_results: Dict) -> Dict[str, Any]:
    """Compare results between full fine-tuning and LoRA."""
    return {
        "full_finetuning": fft_results,
        "peft_lora": lora_results,
        "improvement": {
            "em": lora_results["exact_match"] - fft_results["exact_match"],
            "f1": lora_results["f1"] - fft_results["f1"]
        }
    }