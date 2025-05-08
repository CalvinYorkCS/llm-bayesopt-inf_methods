import torch
import torch.nn as nn
from torch import optim
import tqdm
from transformers import get_scheduler
from contextlib import nullcontext
from llm_bayesopt.inference_method import Inference
from utils.configs import EnsembleConfig
import copy
import math

class EnsembleInference(Inference):
    def __init__(self, ensemble_config: EnsembleConfig, device: str = "cuda", dtype: str = "float32", append_eos: bool = False):
        super().__init__(ensemble_config, device, dtype)
        self.cfg = ensemble_config
        self.append_eos = append_eos
        # Set up amp context, dtype mapping, etc.
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.ctx = (
            nullcontext()
            if device == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        )
        self.enable_grad_scaler = dtype in ["float16", "bfloat16"]
        self.ensemble = []  # Store models in the ensemble

    def reset(self):
        """Reset the ensemble"""
        self.ensemble = []

    def train_single_model(self, model, train_loader):
        """Train a single model in the ensemble"""
        model.train()
        loss_func = nn.MSELoss()

        lora_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "lora" in n
        ]
        head_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "lora" not in n
        ]
        optimizer_lora = optim.AdamW(lora_params, lr=self.cfg.lr_lora, weight_decay=5e-4)
        optimizer_head = optim.AdamW(head_params, lr=self.cfg.lr, weight_decay=5e-4)

        num_training_steps = self.cfg.n_epochs * len(train_loader)
        scheduler_lora = get_scheduler(
            name="linear",
            optimizer=optimizer_lora,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.enable_grad_scaler)

        for _ in tqdm.trange(
            self.cfg.n_epochs, position=1, leave=False, desc="[Training]", colour="blue"
        ):
            for batch in train_loader:
                model.train()
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = model(batch)
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()

                if self.cfg.grad_clip != 0.0:
                    scaler.unscale_(optimizer_lora)
                    torch.nn.utils.clip_grad_norm_(lora_params, self.cfg.grad_clip)

                scaler.step(optimizer_lora)
                scaler.step(optimizer_head)
                scaler.update()
                scheduler_lora.step()
                scheduler_head.step()
                optimizer_lora.zero_grad(set_to_none=True)
                optimizer_head.zero_grad(set_to_none=True)

        # Optional additional fine-tuning for the head
        if self.cfg.finetune_head:
            for n, p in model.named_parameters():
                if "lora" in n:
                    p.requires_grad = False

            optimizer_head = optim.AdamW(head_params, lr=1e-3, weight_decay=5e-4)
            scheduler_head = get_scheduler(
                name="cosine",
                optimizer=optimizer_head,
                num_warmup_steps=0,
                num_training_steps=100 * len(train_loader),
            )

            for _ in tqdm.trange(
                100, position=1, leave=False, desc="[Fine-tuning Head]", colour="green"
            ):
                for batch in train_loader:
                    model.train()
                    labels = batch["labels"].to(self.device, non_blocking=True)

                    with self.ctx:
                        outputs = model(batch)
                        loss = loss_func(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer_head)
                    scaler.update()
                    scheduler_head.step()
                    optimizer_head.zero_grad(set_to_none=True)

            # Restore LoRA parameters to be trainable for next ensemble member
            for n, p in model.named_parameters():
                if "lora" in n:
                    p.requires_grad = True

        model.eval()
        return model

    def train(self, get_model, train_loader):
        """Train all models in the ensemble"""
        self.reset()  # Clear any existing models

        # Train each model in the ensemble
        for i in tqdm.trange(
            self.cfg.n_models, position=0, leave=True, desc="[Ensemble Progress]", colour="cyan"
        ):
            # Get a fresh model instance with different random initialization
            model = get_model().to(self.device)
            
            # If we're using bootstrapped samples, create a new loader
            if self.cfg.bootstrap:
                bootstrap_loader = self.create_bootstrap_loader(train_loader)
                trained_model = self.train_single_model(model, bootstrap_loader)
            else:
                trained_model = self.train_single_model(model, train_loader)
            
            self.ensemble.append(trained_model)

    def create_bootstrap_loader(self, original_loader):
        """Create a bootstrapped version of the dataloader"""
        # This is a simplified implementation - in practice you might want to
        # customize this based on your dataloader structure
        
        # In practice, you would sample with replacement from original_loader's dataset
        # For this implementation, we'll just return the original loader
        # You would typically implement bootstrapping by creating a new DataLoader with
        # a sampler that samples with replacement
        
        return original_loader

    def posterior(self, data):
        """Get the posterior predictive distribution"""
        if not self.ensemble:
            raise RuntimeError("Must call train before posterior")

        with torch.no_grad():
            predictions = []
            for model in self.ensemble:
                model.eval()
                with self.ctx:
                    pred = model(data)
                predictions.append(pred)
            
            # Stack predictions from all models [n_models, batch_size]
            predictions = torch.stack(predictions, dim=0)
            
            # Calculate mean prediction across ensemble
            mean = predictions.mean(dim=0)
            
            # Total variance combines epistemic (between models) and aleatoric uncertainty
            if self.cfg.predict_variance:
                # If models predict both mean and variance, implement accordingly
                # This would need models that output both mean and variance
                pass
            else:
                # Standard deep ensemble with just model disagreement
                variance = predictions.var(dim=0)
                
                # Add observation noise if specified
                if self.cfg.noise_var is not None:
                    variance = variance + self.cfg.noise_var
            
            return torch.distributions.Normal(mean, torch.sqrt(variance))