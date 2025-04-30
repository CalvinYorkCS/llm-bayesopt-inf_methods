import math
import torch
from torch import optim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
from llm_bayesopt.inference_method import Inference  # assuming the base class is defined here
from contextlib import nullcontext
import tqdm
from contextlib import nullcontext
from transformers import get_scheduler
from utils.configs import VIConfig

class VariationalInference(Inference):
    """
    Mean-field Variational Inference using Pyro.
    """
    def __init__(self, vi_config: VIConfig, device: str = "cuda", dtype: str = "float32"):
        super().__init__(vi_config, device, dtype)
        self.cfg = vi_config
        self.num_samples = getattr(vi_config, "num_samples", 20)
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

    def fine_tune(self, model, train_loader):
        loss_func = torch.nn.MSELoss()

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
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.enable_grad_scaler)

        for _ in tqdm.trange(
            self.cfg.n_epochs, position=1, leave=False, desc="[Training]", colour="blue"
        ):
            # for _ in range(self.cfg.n_epochs):
            for batch in train_loader:
                model.train()
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = model(batch)
                    # print(outputs.shape, labels.shape); input()
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

        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False

        optimizer_head = optim.AdamW(head_params, lr=1e-3, weight_decay=5e-4)
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for _ in tqdm.trange(
            100, position=1, leave=False, desc="[Training]", colour="blue"
        ):
            # for _ in range(self.cfg.n_epochs):
            for batch in train_loader:
                model.train()
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = model(batch)
                    # print(outputs.shape, labels.shape); input()
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer_head)
                scaler.update()
                scheduler_head.step()
                optimizer_head.zero_grad(set_to_none=True)

        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        
        return model

    def train(self, get_model, train_loader):
        # fine tune model
        model_for_ft = get_model().to(self.device)

        model_for_ft.train()
        model_for_ft = self.fine_tune(model_for_ft, train_loader)        

        # Clear any previous Pyro parameters
        pyro.clear_param_store()

        # Define the Pyro model and guide
        def model(batch):
            inputs = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Priors: zero-mean normal on all trainable params
            net = model_for_ft
            priors = {}
            for name, param in net.named_parameters():
                if param.requires_grad:
                    prior_std = self.config.prior_std
                    priors[name] = dist.Normal(
                        torch.zeros_like(param),
                        prior_std * torch.ones_like(param)
                    ).to_event(param.dim())

            # Lift module to random_module
            lifted_net = pyro.random_module("net", net, priors)()
            lifted_net.to(self.device)
            lifted_net.train()

            with pyro.plate("data", labels.size(0)):
                preds = lifted_net(batch).squeeze(-1)
                noise = self.config.noise_std
                pyro.sample("obs", dist.Normal(preds, noise).to_event(1), obs=labels) # TODO TEMP added .to_event(1)

        guide = AutoDiagonalNormal(model)
        optimizer = Adam({"lr": self.config.lr})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # Training loop
        for epoch in range(self.config.n_epochs):
            total_loss = 0.0
            for batch in train_loader:
                total_loss += svi.step(batch)
            avg_loss = total_loss / len(train_loader.dataset)
            # Optionally: print(f"Epoch {epoch+1}/{self.config.n_epochs}, ELBO loss: {avg_loss:.4f}")

        # Store guide and model factory for posterior
        self._guide = guide
        self._get_model = lambda: model_for_ft.cpu() # changed from get_model

    def posterior(self, batch):
        # Monte Carlo sampling from the variational posterior
        inputs = batch["input_ids"].to(self.device)
        samples = []
        for _ in range(self.num_samples):
            # Draw a parameter sample
            sampled_model = pyro.random_module(
                "net_pred", self._get_model().to(self.device),
                {
                    name: dist.Normal(
                        torch.zeros_like(param),
                        self.config.prior_std * torch.ones_like(param)
                    ).to_event(param.dim())
                    for name, param in self._get_model().named_parameters()
                    if param.requires_grad
                }
            )()
            sampled_model.eval()
            with torch.no_grad():
                out = sampled_model(batch).squeeze(-1)
            samples.append(out)

        preds = torch.stack(samples, dim=0)  # (S, B)
        mean = preds.mean(0)
        var = preds.var(0) + self.config.noise_std ** 2
        return torch.distributions.Normal(mean, var)

    def reset(self):
        # Clear stored guide so train() restarts fresh
        self._guide = None
        self._get_model = None
        super().reset()