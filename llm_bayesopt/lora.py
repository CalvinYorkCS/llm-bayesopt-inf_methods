import warnings

warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_error()

from torch import nn
import pandas as pd
from .base import LLMBayesOpt
from problems.data_processor import DataProcessor
from inference_method import Inference

from typing import *

class LoRALLMBayesOpt(LLMBayesOpt):
    def __init__(self,
                 get_model: Callable[[], nn.Module],
                 training_set: List[pd.Series],
                 data_processor: DataProcessor,
                 inference: Inference,
                 device="cuda", dtype="float32",
                 append_eos=True):
        self.inference = inference
        super().__init__(get_model, training_set, data_processor, device)  # drop bnn/laplace_config

    def train_model(self):
        train_loader = self.data_processor.get_dataloader(
            pd.DataFrame(self.training_set),
            batch_size=self.inference.config.batch_size,
            shuffle=True,
            append_eos=self.append_eos,
        )
        self.inference.train(self.get_model, train_loader)

    def posterior(self, data):
        return self.inference.posterior(data)

    def condition_on_observations(self, obs):
        self.training_set.append(obs)
        # reset inference for retraining on new data:
        self.inference.reset()
        return LoRALLMBayesOpt(
            get_model=self.get_model,
            training_set=self.training_set,
            data_processor=self.data_processor,
            inference=self.inference,  # same type; will retrain when train_model is next called
            device=self.device, dtype=self.dtype,
            append_eos=self.append_eos,
        )