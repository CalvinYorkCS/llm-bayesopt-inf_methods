from laplace import Laplace
from llm_bayesopt.inference_method import Inference
from laplace.marglik_training import marglik_training
import torch
import torch.nn as nn
from torch import optim
import tqdm
from contextlib import nullcontext
from transformers import get_scheduler
from utils.configs import LaplaceConfig
import math

class LaplaceInference(Inference):
    def __init__(self, laplace_config: LaplaceConfig, device, dtype, append_eos):
        super().__init__(laplace_config, device, dtype)
        self.append_eos = append_eos
        self.cfg = laplace_config
        # set up amp context, dtype mapping, etc.
        self.dtype = dtype
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
        self.append_eos = append_eos

    def train(self, get_model, train_loader):
        self.cfg = self.config
        self.reset()  # drop previous bnn if any

        if self.cfg.marglik_mode == "posthoc":
            self._posthoc_laplace(get_model, train_loader)
        else:
            self._online_laplace(get_model, train_loader)

        # if noise_var was fixed, enforce it
        if self.cfg.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(self.cfg.noise_var)

    def posterior(self, data):
        μ, v = self.bnn(data)
        μ, v = μ.detach(), v.detach().squeeze(-1) + self.bnn.sigma_noise**2
        return torch.distributions.Normal(μ, v)

    def _online_laplace(self, get_model, train_loader):
        la, _, _, _ = marglik_training(
            # Ensure that the base net is re-initialized
            get_model().to(self.device), # changed from self.get_model()...
            train_loader,
            likelihood="regression",
            hessian_structure=self.cfg.hess_factorization,
            n_epochs=self.cfg.n_epochs,
            backend=self.cfg.hessian_backend,
            optimizer_cls=optim.AdamW,
            optimizer_kwargs={"lr": self.cfg.lr},
            scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"T_max": self.cfg.n_epochs * len(train_loader)},
            marglik_frequency=self.cfg.online_marglik_freq,
            prior_structure=self.cfg.prior_prec_structure,
            sigma_noise_fixed=self.cfg.noise_var,
            progress_bar=True,
        )
        self.bnn = la

    def fine_tune(self, model, train_loader):
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

        # So that it's considered by Laplace
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        return model


    def _posthoc_laplace(self, get_model, train_loader):
        model = get_model().to(self.device)  # Ensure that the base net is re-initialized - changed from self.get_model()...

        model.train()
        model = self.fine_tune(model, train_loader)

        model.eval()

        # # Check training perf
        # preds, targets = [], []
        # for batch in train_loader:
        #     preds.append(model(batch))
        #     targets.append(batch['labels'])
        # preds, targets = torch.cat(preds, dim=0).cpu(), torch.cat(targets, dim=0)
        # print(f'Training MSE: {loss_func(preds, targets).item():.3f}')

        if self.cfg.subset_of_weights == "last_layer":
            self.bnn = Laplace(
                model,
                likelihood="regression",
                subset_of_weights=self.cfg.subset_of_weights,
                hessian_structure=self.cfg.hess_factorization,
                sigma_noise=1 if self.cfg.noise_var is None else math.sqrt(self.cfg.noise_var),
                last_layer_name=self.cfg.last_layer_name,
                dict_key_x="input_ids",
            )
        else:
            self.bnn = Laplace(
                model,
                likelihood="regression",
                subset_of_weights=self.cfg.subset_of_weights,
                hessian_structure=self.cfg.hess_factorization,
                sigma_noise=1 if self.cfg.noise_var is None else math.sqrt(self.cfg.noise_var),
                dict_key_x="input_ids",
            )
        # print('Fitting Laplace...')
        self.bnn.fit(train_loader)
 
        # print('Optimizing hyperparams...')
        prior_prec_shapes = {
            "scalar": 1,
            "layerwise": self.bnn.n_layers,
            "diagonal": self.bnn.n_params,
        }
        if self.cfg.noise_var is None:
            # Tune prior precision and observation noise
            log_prior = torch.ones(
                prior_prec_shapes[self.cfg.prior_prec_structure],
                requires_grad=True,
                device=self.device,
            )
            log_sigma = torch.ones(1, requires_grad=True, device=self.device)
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

            for _ in range(self.cfg.posthoc_marglik_iters):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.bnn.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

            self.bnn.prior_precision = log_prior.detach().exp()
            self.bnn.sigma_noise = log_sigma.detach().exp()
        else:
            # Tune only prior precision
            init_prior_prec = torch.ones(
                prior_prec_shapes[self.cfg.prior_prec_structure], device=self.device
            )
            self.bnn.optimize_prior_precision(
                n_steps=self.cfg.posthoc_marglik_iters, init_prior_prec=init_prior_prec
            )
            # print(self.bnn.prior_precision)
        # print('Done!')