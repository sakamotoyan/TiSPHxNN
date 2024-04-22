import torch
import taichi as ti

def select_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available. Using CUDA...")
    # If CUDA is not available, check for MPS (available on macOS with Apple Silicon)
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("CUDA not available, but MPS is available. Using MPS...")
    else:
        device = 'cpu'
        print("Neither CUDA nor MPS is available. Using CPU...")

    return device

def select_ti_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = ti.cuda
        print("CUDA is available. Using CUDA...")
    # If CUDA is not available, check for MPS (available on macOS with Apple Silicon)
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("CUDA not available, but MPS is available. Using MPS...")
    else:
        device = 'cpu'
        print("Neither CUDA nor MPS is available. Using CPU...")

    return device