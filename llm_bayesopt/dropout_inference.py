import torch
import torch.nn as nn
from torch import optim
import tqdm
from transformers import get_scheduler
from contextlib import nullcontext
from llm_bayesopt.inference_method import Inference
from utils.configs import MCDropoutConfig

class MCDropoutInference(Inference):
    def __init__(self, mcdropout_config, device, dtype, append_eos):
        super().__init__(mcdropout_config, device, dtype)
        self.cfg = mcdropout_config
        self.append_eos = append_eos
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
            if device != "cpu"
            else nullcontext()
        )
        self.enable_grad_scaler = dtype in ["float16", "bfloat16"]

    def reset(self):
        self.bnn = None

    def train(self, get_model, train_loader):
        self.reset()
        model = get_model().to(self.device)
        model.train()  # Keep dropout active during training

        loss_func = nn.MSELoss()
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
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

        self.bnn = model

    def posterior(self, data):
        self.bnn.train()  # Enable dropout during inference
        samples = []
        for _ in range(self.cfg.n_samples):
            with self.ctx:
                with torch.no_grad():
                    pred = self.bnn(data)
                samples.append(pred)
        samples = torch.stack(samples, dim=0)  # [n_samples, batch_size]
        mean = samples.mean(dim=0)  # [batch_size]
        variance = samples.var(dim=0)  # [batch_size]
        # Add observation noise if specified
        if self.cfg.noise_var is not None:
            variance = variance + self.cfg.noise_var
        return torch.distributions.Normal(mean, variance.sqrt())