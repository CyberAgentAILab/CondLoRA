import os
import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.linalg import svd

from models.load_model import load_pretrained_model
from visualization.visualization import plot_heatmap


def cal_phai(matrix1: torch.tensor, matrix2: torch.tensor, i: int, j: int, left_or_right: str) -> float:
    u1, s1, v1 = svd(matrix1, full_matrices=True)
    u2, s2, v2 = svd(matrix2, full_matrices=True)
    if left_or_right == "left":  # use left singular vectors        
        f_norm = torch.norm(torch.mm(u1[:, :i].T, u2[:, :j])) ** 2
    elif left_or_right == "right":  # use right singular vectors
        v1, v2 = v1.T, v2.T
        f_norm = torch.norm(torch.mm(v1[:, :i].T, v2[:, :j])) ** 2

    return f_norm / min(i, j)


def phai(linears: List[torch.tensor], i: int, j: int, left_or_right: str, save_path: str = None):
    phai_scores = np.ones((len(linears), len(linears)))
    random_phai_scores = np.ones((len(linears), len(linears)))

    random_matrices = []
    for _ in range(len(linears)):
        random_matrix = torch.randn(linears[0].shape[0], linears[0].shape[1])
        u, s, v = svd(random_matrix, full_matrices=True)
        if left_or_right == "left":
            random_matrices.append(u)
        elif left_or_right == "right":
            random_matrices.append(v.T)

    for n1 in range(len(linears)):
        linear1 = linears[n1]
        random_matrix1 = random_matrices[n1]
        u, s, v = svd(linear1)
        if i == j:
            start = n1
        else:
            start = 0
        for n2 in range(start, len(linears)):
            linear2 = linears[n2]
            random_matrix2 = random_matrices[n2]
            phai_score = cal_phai(linear1, linear2, i, j, left_or_right)

            random_phai_score = torch.norm(torch.mm(random_matrix1[:, :i].T, random_matrix2[:, :j])) ** 2
            random_phai_score = random_phai_score / min(i, j)
            
            phai_scores[n1, n2] = phai_score.item()
            random_phai_scores[n1, n2] = random_phai_score.item()
            if i == j:
                phai_scores[n2, n1] = phai_score
                random_phai_scores[n2, n1] = random_phai_score
    
    if save_path:
        vmax = 1.0
        plot_heatmap(phai_scores, fmt=".2f", vmin=0.0, vmax=vmax, save_path=f"{save_path}.png")
        plot_heatmap(random_phai_scores, fmt=".2f", vmin=0.0, vmax=vmax, save_path=f"{save_path}.random.png")
    return phai_scores


def get_linears(lora_model):
    linears = defaultdict(list)
    for n, lora_layer in enumerate(lora_model.base_model.model.roberta.encoder.layer):
        lora_modules = {"q": lora_layer.attention.self.query, "v": lora_layer.attention.self.value}
        for module_name in ["q", "v"]:
            w0 = lora_modules[module_name].weight.detach().cpu()

            lora_A = lora_modules[module_name].lora_A.default.weight.detach().cpu().T # shape = r, dim1
            lora_B = lora_modules[module_name].lora_B.default.weight.detach().cpu()  # shape = r, dim2
            inv_w0 = torch.linalg.inv(w0)
            linear_w0_to_A = torch.mm(inv_w0, lora_A)
            linear_w0_to_B = torch.mm(inv_w0, lora_B)
            linear_w0_to_BA = torch.mm(inv_w0, torch.mm(lora_B, lora_A.T))

            linears[f"{module_name}_A"].append(linear_w0_to_A)
            linears[f"{module_name}_B"].append(linear_w0_to_B)
            linears[f"{module_name}_BA"].append(linear_w0_to_BA)
    
    return linears 


def cal_linear_similarity(lora_model, i: int, j: int, left_or_right: str, save_dir: str):
    linears = get_linears(lora_model=lora_model)
    for module_name in ["q", "v"]:
        for lora_name in ["A", "B", "BA"]:
            target_linears = linears[f"{module_name}_{lora_name}"]

            os.makedirs(f"{save_dir}/phai/i={i}-j={j}-{left_or_right}", exist_ok=True)
            _ = phai(target_linears, i, j, left_or_right=left_or_right, save_path=f"{save_dir}/phai/i={i}-j={j}-{left_or_right}/{module_name}.{lora_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path",
    )
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--i", type=int)
    parser.add_argument("--j", type=int)
    parser.add_argument("--left_or_right", type=str)
    args = parser.parse_args()

    lora_model, _ = load_pretrained_model(model_path=args.model_path, num_labels=args.num_labels)
    cal_linear_similarity(lora_model=lora_model, i=args.i, j=args.j, left_or_right=args.left_or_right, save_dir=args.model_path)
