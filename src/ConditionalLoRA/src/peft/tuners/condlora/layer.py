import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class LoraLayer(BaseTunerLayer):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_x_scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.linear_for_lora_A = nn.ModuleDict({})
        self.linear_for_lora_B = nn.ModuleDict({})
        self.linear_for_lora_x = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha, 
        lora_dropout,
        linear_for_lora_A,
        linear_for_lora_B,
        linear_for_lora_x,
        lora_x_scaling,
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if linear_for_lora_x:
            assert 0.0 < lora_x_scaling <= 1.0, "Please set lora_x_scaling lora_x_scaling in the range of 0.0 to 1"
        else:
            assert lora_x_scaling == 0.0, "Please set lora_x_scaling to 0.0 if you don't use x for lora"
        # Actual trainable parameters
        if r > 0:
            self.linear_for_lora_A[adapter_name] = linear_for_lora_A
            self.linear_for_lora_B[adapter_name] = linear_for_lora_B
            self.linear_for_lora_x[adapter_name] = linear_for_lora_x
            self.scaling[adapter_name] = lora_alpha / r
            self.lora_x_scaling[adapter_name] = lora_x_scaling
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.linear_for_lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.linear_for_lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.linear_for_lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        linear_for_lora_A: nn.Linear,
        linear_for_lora_B: nn.Linear,
        linear_for_lora_x: nn.Linear = None,
        use_x: str = "none",
        lora_x_scaling: float = 0.5,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.use_x = use_x

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            linear_for_lora_A=linear_for_lora_A,
            linear_for_lora_B=linear_for_lora_B,
            linear_for_lora_x=linear_for_lora_x,
            lora_x_scaling=lora_x_scaling,
        )
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self) -> None:
        # TODO: implement merge operation for ConditionalLoRA
        if self.active_adapter not in self.linear_for_lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            scaling = self.scaling[self.active_adapter]
            self.weight.data += scaling * self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self) -> None:
        if self.active_adapter not in self.linear_for_lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            scaling = self.scaling[self.active_adapter]
            self.weight.data -= scaling * self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
            transpose(
                self.linear_for_lora_B[adapter](self.weight) @ self.linear_for_lora_A[adapter](self.weight).T,
                self.fan_in_fan_out,
            )
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.linear_for_lora_A.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if (self.r[self.active_adapter] > 0) and self.merged:
                self.unmerge()
            result = self._linear(x)
        elif (self.r[self.active_adapter] == 0) or self.merged:
            result = self._linear(x)
        else:
            linear_for_lora_A = self.linear_for_lora_A[self.active_adapter] # shape = (dim, r)
            linear_for_lora_B = self.linear_for_lora_B[self.active_adapter] # shape = (dim, r)
            linear_for_lora_x = self.linear_for_lora_x[self.active_adapter] # if use_x == "type1" then shape = (dim, r), elif use_x == "type2" then shape = (dim, 1)
 
            lora_A = linear_for_lora_A(self.weight).T # shape = (r, dim)
            lora_B = linear_for_lora_B(self.weight) # shape = (dim, r)
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]
            lora_x_scaling = self.lora_x_scaling[self.active_adapter]


            x = x.to(self.linear_for_lora_A[self.active_adapter].weight.dtype)
            x = dropout(x)
            result = self._linear(x)
            
            if self.use_x == "type1":
                x_ =  lora_x_scaling * linear_for_lora_x(x)
            elif self.use_x == "type2":
                # dot operation between linear_for_lora_x and [BOS] hidden states of each example
                lora_x = linear_for_lora_x(torch.unsqueeze(x[:, 0, :], -1)) # shape = (batch_size, dim, dim)
                x_ = torch.bmm(x, lora_x) * lora_x_scaling * scaling

            x = F.linear(x, lora_A)
            if self.use_x == "type1":
                x += x_
            x = F.linear(x, lora_B)

            result += x * scaling
            if self.use_x == "type2":
                result += x_

        result = result.to(previous_dtype)
        return result