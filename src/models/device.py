def get_device():
    """Picks the best available torch device (MPS, CUDA, or CPU)."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
