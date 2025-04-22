from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Callable

class Inference(ABC):
    def __init__(self, config, device: str, dtype: str):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.bnn = None

    @abstractmethod
    def train(self, get_model: Callable[[], nn.Module], train_loader):
        """Fit whatever approximate posterior or surrogate model is needed."""
        pass

    @abstractmethod
    def posterior(self, inputs: torch.Tensor):
        """Given inputs, return a torch.distributions object with mean/variance."""
        pass

    def reset(self):
        """Clear any existing state so that future train() starts fresh."""
        self.bnn = None