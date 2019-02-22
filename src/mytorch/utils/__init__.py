import torch

use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

device = torch.device("cuda" if use_cuda else "cpu")
