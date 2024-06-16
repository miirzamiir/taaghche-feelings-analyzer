import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric


class TranformerClassification:
    model_names= [
        "xlm-roberta-base",
        "bert-base-multilingual-cased"
    ]
    
    def __init__(self, model_name: str, **keywords) -> None:
        self._model_name = model_name
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Freeze all layers except the last few
    def freeze_model_layers(model, model_name):
        if "bert" in model_name.lower():
            # Freeze all layers except the classifier and the last layer
            for name, param in model.named_parameters():
                if not any(n in name for n in ['classifier', 'pooler', 'encoder.layer.11', 'encoder.layer.10']):
                    param.requires_grad = False
        elif "roberta" in model_name.lower():
            # Freeze all layers except the classifier and the last layer
            for name, param in model.named_parameters():
                if not any(n in name for n in ['classifier', 'pooler', 'layer.11', 'layer.10']):
                    param.requires_grad = False
        else:
            raise ValueError(f"Model {model_name} not supported for automatic layer freezing.")
    
    def tokenize_function(self, examples):
        return self._tokenizer(examples['text'], padding="max_length", truncation=True)

dataset = load_dataset('csv', data_files={'train': 'path/to/train.csv', 'test': 'path/to/test.csv'})
