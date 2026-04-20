import sys
import torch

model = torch.load(sys.argv[1], map_location="cpu")
print(model)

