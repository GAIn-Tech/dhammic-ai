import torch

def compute_overlap(sdr1, sdr2):
    # Pack bits into uint8
    # PyTorch doesn't have packbits, numpy does.
    # But we can just use bool tensors
    return torch.bitwise_and(sdr1.bool(), sdr2.bool()).sum(dim=-1)

sdr1 = torch.randint(0, 2, (2, 10, 256))
sdr2 = torch.randint(0, 2, (2, 10, 256))
print(compute_overlap(sdr1, sdr2).shape)
