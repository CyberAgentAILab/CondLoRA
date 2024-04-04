TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def tokenize_function(examples, task, tokenizer, max_length):
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    tokenized_inputs = tokenizer(
        *args, padding="max_length", truncation=True, max_length=max_length
    )

    return tokenized_inputs
