from typing import Tuple

import torch 
from transformers import AutoModelForSequenceClassification
from ConditionalLoRA.src.peft import PeftModel, PeftConfig, get_peft_model


def load_initial_model(
    model_name_or_path: str, 
    lora_type: str, 
    r: int, 
    lora_alpha: int, 
    lora_dropout: float,
    task_type: str,
    lora_x_scaling: float = 0.0,
    num_lables: int = 2,
) -> PeftModel:
    assert task_type == "SEQ_CLS", f"Not implemented task type [{task_type}]"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_lables)
    
    if lora_type == "lora":
        from peft import LoraConfig 
        lora_config =  LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    elif lora_type == "adalora":
        from peft import AdaLoraConfig 
        lora_config =  AdaLoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    elif "conditional_lora" in lora_type:
        from peft import ConditionalLoraConfig
        if "conditional_lora_x" in lora_type:
            assert lora_x_scaling > 0.0, f"Plese set --lora_x_scaling."
        else:
            lora_x_scaling = 0.0
            use_x = "none"
        
        if lora_type == "conditional_lora_x_type1":
            use_x = "type1"
        elif lora_type == "conditional_lora_x_type2":
            use_x = "type2"
        

        lora_config = ConditionalLoraConfig(
            task_type="SEQ_CLS",
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_x=use_x,
            lora_x_scaling=lora_x_scaling,
        )

    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()

    return model


def load_pretrained_model(model_path: str, num_labels: int) -> Tuple[PeftModel, PeftConfig]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PeftConfig.from_pretrained(model_path)
    model_name_or_path = config.base_model_name_or_path
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = PeftModel.from_pretrained(model, model_path, is_trainable=False).to(device)
    return model, config