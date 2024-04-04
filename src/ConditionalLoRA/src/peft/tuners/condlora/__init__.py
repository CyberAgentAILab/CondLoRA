from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .config import ConditionalLoraConfig
from .gptq import QuantLinear
from .layer import Linear, LoraLayer
from .model import ConditionalLoraModel


__all__ = ["ConditionalLoraConfig", "Conv2d", "Embedding", "LoraLayer", "Linear", "ConditionalLoraModel", "QuantLinear"]


if is_bnb_available():
    from .bnb import Linear8bitLt

    __all__ += ["Linear8bitLt"]

if is_bnb_4bit_available():
    from .bnb import Linear4bit

    __all__ += ["Linear4bit"]
