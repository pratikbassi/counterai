from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def fit_temperature_scaling(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    max_iter: int = 50,
    lr: float = 0.05,
) -> float:
    """
    Fit a single temperature T on validation logits so softmax(logits/T) is better calibrated.

    Tradeoff: extra pass over val data; T is stored in checkpoint for inference.
    """

    model.eval()
    nll = nn.CrossEntropyLoss()

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
            logits_list.append(logits.float().detach())
            labels_list.append(labels)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        t = log_t.exp().clamp(min=1e-3, max=100.0)
        loss = nll(all_logits / t, all_labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().clamp(min=1e-3, max=100.0).detach().item())
    return t
