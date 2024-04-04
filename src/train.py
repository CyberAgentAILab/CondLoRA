import argparse
import random

import numpy as np
import torch
from data.load_dataset import load_dataset
from data.preprocess import tokenize_function
from datasets import load_metric
from models.load_model import load_initial_model
from torch.optim import AdamW
from trainer import Trainer
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments


def torch_fix_seed(seed: int = 42) -> None:
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path",
    )
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--early_stopping", type=int)
    parser.add_argument("--evaluation_steps", type=int, default=1500)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_cycles", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--setting", type=str)
    parser.add_argument("--lora_type", type=str, default="lora")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_x_scaling", type=float, default=0.0)
    args = parser.parse_args()

    print(args)
    torch_fix_seed(args.seed)

    assert args.lora_type in [
        "lora",
        "adalora",
        "conditional_lora",
        "conditional_lora_x_type1",
        "conditional_lora_x_type2",
    ], f"Plese select --lora_type from lora, adalora, and conditional_lora"
    # load dataset, model, tokenizer
    if args.setting == "lora":
        from datasets import concatenate_datasets, load_dataset

        dataset = load_dataset("glue", args.task)
        label_set = set(dataset["train"]["label"])

        if args.mode == "debug":
            dataset["train"] = dataset["train"].select(range(100))
            dataset["test"] = dataset["test"].select(range(100))
            dataset["validation"] = dataset["validation"].select(range(100))

        if args.task == "mnli":
            dataset["validation"] = concatenate_datasets(
                [dataset["validation_matched"], dataset["validation_mismatched"]]
            )

        evaluation_strategy = "epoch"
    else:
        dataset, label_set = load_dataset(task=args.task, mode=args.mode)
        iteration_per_epoch = round(len(dataset["train"]) / args.batch_size)
        if iteration_per_epoch >= args.evaluation_steps:
            evaluation_strategy = "steps"
        else:
            evaluation_strategy = "epoch"

    model = load_initial_model(
        model_name_or_path=args.model_name_or_path,
        lora_type=args.lora_type,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="SEQ_CLS",
        lora_x_scaling=args.lora_x_scaling,
        num_lables=1 if args.task == "stsb" else len(label_set),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # tokenize dataset
    dataset = dataset.map(
        lambda example: tokenize_function(
            example, task=args.task, tokenizer=tokenizer, max_length=args.max_length
        )
    )
    dataset = dataset.remove_columns(
        [
            x
            for x in dataset["train"].features
            if x not in ["input_ids", "attention_mask", "label"]
        ]
    )

    metric_for_best_model = "accuracy"
    if args.task == "stsb":
        metric_for_best_model = "pearson"
    elif args.task == "cola":
        metric_for_best_model = "matthews_correlation"

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy=evaluation_strategy,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy=evaluation_strategy,
        save_total_limit=2,
        label_names=["labels"],
        remove_unused_columns=False,
        save_steps=args.evaluation_steps,
        eval_steps=None if evaluation_strategy == "epoch" else args.evaluation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model=metric_for_best_model,
        load_best_model_at_end=False if args.setting == "lora" else True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.max_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    # definne compute metrics
    metric = load_metric("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.squeeze(logits) if args.task == "stsb" else np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_scheduler_type == "cosine_with_restarts":
        assert (
            args.num_cycles
        ), f"Please set --num_cycles if you use cosine_with_restarts for scheduler"
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

        num_training_steps = round(
            args.max_epochs * len(dataset["train"]) / args.batch_size
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            num_cycles=args.num_cycles,
        )
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=(
            None
            if args.setting == "lora" or not args.early_stopping
            else [EarlyStoppingCallback(early_stopping_patience=args.early_stopping)]
        ),
        optimizers=(optimizer, scheduler),
    )

    # training
    trainer.train()
