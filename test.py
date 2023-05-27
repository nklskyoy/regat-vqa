import torch

device = torch.device("cuda")
torch.Tensor(1).to(device)

if not torch.cuda.is_available():
    raise ValueError("CUDA is not available," +
                        "this code currently only support GPU.")
else:
    print("Welcome to pytorch")
