from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

class Ensemble(ABC, torch.nn.Module):
    @abstractmethod
    # return dictionay of the predictions and loss
    # predictions should be identical to the output of the model in models [ex: src/models/bert.py]
    def forward(self, batch, train=False) -> Tuple[Dict[str, Any], Any]:
        pass