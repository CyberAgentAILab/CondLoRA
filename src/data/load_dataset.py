from typing import Tuple

from datasets import DatasetDict, concatenate_datasets
from datasets import load_dataset as _load_dataset


def load_dataset(task: str, mode: str = "train") -> Tuple[DatasetDict, set]:
    if task == "mnli":
        return load_mnli_dataset(mode=mode)
    
    ds = _load_dataset("glue", task)
    ds["test"] = ds["validation"]

    valid_rate = 0.1
    valid_num = round(len(ds["train"]) * valid_rate)
    ds["train"] = ds["train"].shuffle(seed=1)
    trainset = ds["train"].select(range(valid_num, len(ds["train"])))
    validset = ds["train"].select(range(valid_num))
    ds["train"] = trainset
    ds["validation"] = validset
    label_set = set(ds["train"]["label"])

    if mode == "debug":
        ds["train"] = ds["train"].select(range(100))
        ds["test"] = ds["test"].select(range(100))
        ds["validation"] = ds["validation"].select(range(100))

    return ds, label_set
    

def load_mnli_dataset(mode: str = "train") -> Tuple[DatasetDict, set]:
    ds = _load_dataset("glue", "mnli")
    ds["test_matched"] = ds["validation_matched"]
    ds["test_mismatched"] = ds["validation_mismatched"]
    ds["test"] = concatenate_datasets([ds["validation_matched"], ds["validation_mismatched"]])
    ds.pop("validation_matched")
    ds.pop("validation_mismatched")

    valid_rate = 0.1
    valid_num = round(len(ds["train"]) * valid_rate)

    ds["train"] = ds["train"].shuffle(seed=1)
    trainset = ds["train"].select(range(valid_num, len(ds["train"])))
    validset = ds["train"].select(range(valid_num))
    ds["train"] = trainset
    ds["validation"] = validset
    label_set = set(ds["train"]["label"])

    if mode == "debug":
        ds["train"] = ds["train"].select(range(100))
        ds["test_matched"] = ds["test_matched"].select(range(100))
        ds["test_mismatched"] = ds["test_mismatched"].select(range(100))
        ds["validation"] = ds["validation"].select(range(100))
        ds["test"] = ds["test"].select(range(100))

    return ds, label_set
