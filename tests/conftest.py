"""Match the study's single-threaded CPU policy for fast deterministic tests."""
import torch
torch.set_num_threads(1)
