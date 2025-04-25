import math
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
from llm_bayesopt.inference_method import Inference  # assuming the base class is defined here
from contextlib import nullcontext

class VariationalInference(Inference):
    """
    Mean-field Variational Inference using Pyro.
    """
    def __init__(self, vi_config, device: str = "cuda", dtype: str = "float32"):
        super().__init__(vi_config, device, dtype)
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

    def train(self, get_model, train_loader):
        # fine tune model
        model_for_ft = get_model().to(self.device)
        optimizer_ft = torch.optim.AdamW(model_for_ft.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        model_for_ft.train()
        for epoch in range(self.config.ft_epochs):
            for batch in train_loader:
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                with self.ctx:
                    preds = model_for_ft(inputs).squeeze(-1)
                    loss = loss_fn(preds, labels)
                optimizer_ft.zero_grad()
                loss.backward()
                optimizer_ft.step()

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
                preds = lifted_net(inputs).squeeze(-1)
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
                out = sampled_model(inputs).squeeze(-1)
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