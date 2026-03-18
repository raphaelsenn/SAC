import torch
import numpy as np


def ensure_tensorf32(
    s: np.ndarray | torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    if isinstance(s, torch.Tensor):
        s = s.to(device=device, dtype=torch.float32)
    elif isinstance(s, np.ndarray):
        s = torch.as_tensor(s, dtype=torch.float32, device=device)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(s)}")

    return s