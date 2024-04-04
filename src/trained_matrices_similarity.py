import argparse
import math

import numpy as np
# from numpy.linalg import svd
import torch
from torch.linalg import svd

from models.load_model import load_pretrained_model


def phai_f(matrix1: torch.tensor, matrix2: torch.tensor, i: int, j: int, l_or_r: str) -> float:
    u1, _, v1 = svd(matrix1, full_matrices=True)
    u2, _, v2 = svd(matrix2, full_matrices=True)
    if l_or_r == "left":  # use left singular vectors        
        f_norm = torch.norm(torch.mm(u1[:, :i].T, u2[:, :j])) ** 2
        random_u2 = torch.nn.init.orthogonal_(torch.empty(u2.shape[0], u2.shape[1]))
        rand_f_norm = torch.norm(torch.mm(u1[:, :i].T, random_u2[:, :j])) ** 2
    elif l_or_r == "right":  # use right singular vectors
        v1, v2 = v1.T, v2.T
        f_norm = torch.norm(torch.mm(v1[:, :i].T, v2[:, :j])) ** 2
        random_v1 = torch.nn.init.orthogonal_(torch.empty(v1.shape[0], v1.shape[1]))
        random_v2 = torch.nn.init.orthogonal_(torch.empty(v2.shape[0], v2.shape[1]))
        rand_f_norm = torch.norm(torch.mm(v1[:, :i].T, random_v2[:, :j])) ** 2
    
    return f_norm / min(i, j), rand_f_norm / min(i, j)


def get_condlora_param(module):
    initial_w = module.weight.detach().cpu()
    lora_A = module.linear_for_lora_A.default.to("cpu")(initial_w).T
    lora_B = module.linear_for_lora_B.default.to("cpu")(initial_w)
    return lora_A.detach().cpu(), lora_B.detach().cpu()


def cal_phai(lora_model, condlora_model, i: int, j: int):
    all_phai_A = 0
    all_phai_rand_A = 0
    all_phai_B = 0
    all_phai_rand_B = 0
    all_phai_delta = 0
    all_phai_rand_delta = 0
    for n, (lora_layer, condlora_layer) in enumerate(zip(lora_model.base_model.model.roberta.encoder.layer, condlora_model.base_model.model.roberta.encoder.layer)):
        lora_modules = {"q": lora_layer.attention.self.query, "v": lora_layer.attention.self.value}
        condlora_modules = {"q": condlora_layer.attention.self.query, "v": condlora_layer.attention.self.value}
        for module_name in ["q", "v"]:
            lora_A = lora_modules[module_name].lora_A.default.weight.detach().cpu()
            lora_B = lora_modules[module_name].lora_B.default.weight.detach().cpu()
            condlora_A, condlora_B = get_condlora_param(condlora_modules[module_name])

            # calculate normalized subspace similarity (see section7 of the LoRA paper)
            phai_A, phai_rand_A = phai_f(matrix1=lora_A, matrix2=condlora_A, i=i, j=j, l_or_r="right")
            phai_B, phai_rand_B = phai_f(matrix1=lora_B, matrix2=condlora_B, i=i, j=j, l_or_r="left")
            phai_delta, phai_rand_delta = phai_f(
                matrix1=torch.mm(lora_B, lora_A),
                matrix2=torch.mm(condlora_B, condlora_A),
                i=i,
                j=j,
                l_or_r="left" if lora_B.shape[0] >= lora_B.shape[1] else "right"
            )
            all_phai_A += phai_A
            all_phai_rand_A += phai_rand_A
            all_phai_B += phai_B
            all_phai_rand_B += phai_rand_B
            all_phai_delta += phai_delta
            all_phai_rand_delta += phai_rand_delta

            print(f"Layer{n}, {module_name}, A", f"phai = {phai_A} ({phai_rand_A})")
            print(f"Layer{n}, {module_name}, B", f"phai = {phai_B} ({phai_rand_B})")
            print(f"Layer{n}, {module_name}, delta", f"{phai_delta} ({phai_rand_delta})")
        
        print()

    print(
        f"Average A", 
        f"phai = {all_phai_A / ((n + 1) * 2)} ({all_phai_rand_A / ((n + 1) * 2)})", 
    )
    print(
        f"Average B", 
        f"phai = {all_phai_B / ((n + 1) * 2)} ({all_phai_rand_B / ((n + 1) * 2)})",
    )
    print(
        f"Average delta",
        f"phai = {all_phai_delta / ((n + 1) * 2)} ({all_phai_rand_delta / ((n + 1) * 2)})",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_path",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--condlora_path",
        type=str,
        help="model path",
    )
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--i", type=int)
    parser.add_argument("--j", type=int)
    args = parser.parse_args()

    lora_model, _ = load_pretrained_model(model_path=args.lora_path, num_labels=args.num_labels)
    condlora_model, _ = load_pretrained_model(model_path=args.condlora_path, num_labels=args.num_labels)
    cal_phai(lora_model=lora_model, condlora_model=condlora_model, i=args.i, j=args.j)