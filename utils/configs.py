from laplace.curvature import AsdlGGN, CurvatureInterface
from typing import *
from enum import Enum


class LLMFeatureType(Enum):
    LAST_TOKEN = 1
    FIRST_TOKEN = 2
    AVERAGE = 3


class LaplaceConfig:
    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-3,
        lr_lora: float = 3e-4,
        wd: float = 0.01,
        grad_clip: float = 0.0,
        n_epochs: int = 50,
        marglik_mode: str = "posthoc",
        noise_var: Union[float, None] = None,
        subset_of_weights: str = "all",
        hess_factorization: str = "diag",
        prior_prec_structure: str = "layerwise",
        posthoc_marglik_iters: int = 200,
        online_marglik_freq: int = 5,
        hessian_backend: CurvatureInterface = AsdlGGN,
        last_layer_name: str = "base_model.model.head.modules_to_save.default.2",
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.lr_lora = lr_lora
        self.wd = wd
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.marglik_mode = marglik_mode
        self.noise_var = noise_var
        self.subset_of_weights = subset_of_weights
        self.hess_factorization = hess_factorization
        assert prior_prec_structure in ["scalar", "layerwise", "diagonal"]
        self.prior_prec_structure = prior_prec_structure
        self.posthoc_marglik_iters = posthoc_marglik_iters
        self.online_marglik_freq = online_marglik_freq
        self.hessian_backend = hessian_backend
        self.last_layer_name = last_layer_name

class VIConfig:
    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-3,
        lr_lora: float = 3e-4,
        wd: float = 0.01,
        grad_clip: float = 0.0,
        n_epochs: int = 50,
        n_samples: int = 10,
        kl_scale: float = 1.0,
        init_logvar: float = -5.0,
        prior_std: float = 1.0,
        inference_method: str = "mean-field",
        last_layer_name: str = "base_model.model.head.modules_to_save.default.2",
        ft_epochs: int = 10,  # Added: Number of fine-tuning epochs
        noise_std: float = 0.1,  # Added: Standard deviation of observation noise
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.lr_lora = lr_lora
        self.wd = wd
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.kl_scale = kl_scale
        self.init_logvar = init_logvar
        self.prior_std = prior_std
        self.inference_method = inference_method
        self.last_layer_name = last_layer_name
        self.ft_epochs = ft_epochs  # Added
        self.noise_std = noise_std  # Added

class MCDropoutConfig:
    def __init__(
        self,
        n_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        lr_lora: float = 3e-4,
        grad_clip: float = 1.0,
        n_samples: int = 20,  # Number of MC samples for inference
        noise_var: float = 0.001,  # Observation noise variance
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_lora = lr_lora
        self.grad_clip = grad_clip
        self.n_samples = n_samples
        self.noise_var = noise_var

