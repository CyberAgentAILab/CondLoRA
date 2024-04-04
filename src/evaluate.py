import argparse

import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from trainer import Trainer
from models.load_model import load_pretrained_model
from data.load_dataset import load_dataset
from data.preprocess import tokenize_function
from datasets import load_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path",
    )
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    # load dataset, model, tokenizer
    dataset, label_set = load_dataset(task=args.task, mode=args.mode)
    model, config = load_pretrained_model(
        model_path=args.model_path,
        num_labels=1 if args.task == "stsb" else len(label_set)
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # tokenize dataset
    splits = ["validation", "test"]
    if args.task == "mnli":
        splits += ["test_matched", "test_mismatched"]
    for split in splits:
        dataset[split] = dataset[split].map(
            lambda example: tokenize_function(example, task=args.task, tokenizer=tokenizer, max_length=args.max_length)
        )
        dataset[split] = dataset[split].remove_columns(
            [x for x in dataset[split].features if x not in ["input_ids", "attention_mask", "label"]]
        )

    training_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=0,
        label_names=["labels"],
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model="acc",
        load_best_model_at_end=True,
    )

    metric = load_metric("glue", args.task)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.squeeze(logits) if args.task == "stsb" else np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    metrics = trainer.evaluate(dataset["validation"])
    print("Validation scores")
    print(metrics)

    metrics = trainer.evaluate(dataset["test"])
    print("Test scores")
    print(metrics)

    if args.task == "mnli":
        mnli_scores = trainer.evaluate(dataset["test_matched"])
        mnli_mm_scores = trainer.evaluate(dataset["test_mismatched"])
        print("Test_matched scores")
        print(mnli_scores)
        print("Test_mismatched scores")
        print(mnli_mm_scores)