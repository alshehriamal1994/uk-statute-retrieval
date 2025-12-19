import argparse
import json
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class QueryPassageExample:
    query: str
    passage: str
    label: float

class QPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: QueryPassageExample = self.data[idx]
        enc = self.tokenizer(
            item.query,
            item.passage,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(item.label, dtype=torch.float)
        return enc

def load_regression_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            data.append(
                QueryPassageExample(
                    query=o["query"],
                    passage=o["passage"],
                    label=float(o.get("teacher_score", 0.0)),
                )
            )
    return data

def main():
    parser = argparse.ArgumentParser(description="Train Distilled ModernBERT for Legal Reranking")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL (query, passage, teacher_score)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--model_name", type=str, default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(42)

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1, # Regression
        trust_remote_code=True,
    )

    print(f"Loading data from {args.data} ...")
    all_data = load_regression_data(args.data)
    print(f"Total examples: {len(all_data)}")

    # 90/10 Split
    split_idx = int(0.9 * len(all_data))
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    train_dataset = QPDataset(train_data, tokenizer, max_length=args.max_length)
    eval_dataset = QPDataset(eval_data, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="pearson"
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        pearson = np.corrcoef(preds, labels)[0, 1]
        try:
            from scipy.stats import spearmanr
            spearman = spearmanr(preds, labels).correlation
        except: spearman = 0.0
        return {"pearson": float(pearson), "spearman": float(spearman)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
