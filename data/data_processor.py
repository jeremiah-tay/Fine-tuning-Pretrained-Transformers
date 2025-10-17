# data/data_processor.py
from transformers import AutoTokenizer
from typing import Dict, Any, List
import numpy as np
import collections
from tqdm.auto import tqdm

def preprocess_qa_examples(examples: Dict[str, Any], 
                          tokenizer: AutoTokenizer,
                          max_length: int = 384,
                          stride: int = 128) -> Dict[str, Any]:
    """Preprocess QA examples for training."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = tokenized_examples["offset_mapping"]
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer_start_char = examples["answer_start"][sample_index]
        answer_end_char = examples["answer_end"][sample_index]
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Find context start and end tokens
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start_token = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end_token = idx - 1
        
        # Check if answer is in context
        if not (offsets[context_start_token][0] <= answer_start_char and 
                offsets[context_end_token][1] >= answer_end_char):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Find start token
            token_start_index = context_start_token
            while (token_start_index <= context_end_token and 
                   offsets[token_start_index][0] < answer_start_char):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index)
            
            # Find end token
            token_end_index = context_end_token
            while (token_end_index >= context_start_token and 
                   offsets[token_end_index][1] > answer_end_char):
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index)
    
    return tokenized_examples

def postprocess_qa_predictions(examples: List[Dict], 
                              features: List[Dict], 
                              raw_predictions: tuple,
                              tokenizer: AutoTokenizer,
                              n_best_size: int = 20, 
                              max_answer_length: int = 30) -> Dict[str, str]:
    """Post-process QA predictions."""
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = collections.OrderedDict()
    
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (start_index >= len(offset_mapping) or 
                        end_index >= len(offset_mapping) or
                        offset_mapping[start_index] is None or 
                        offset_mapping[end_index] is None or
                        end_index < start_index or 
                        end_index - start_index + 1 > max_answer_length):
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                    })
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
    
    return predictions