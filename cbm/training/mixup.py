import torch
import torch.nn.functional as F


def mixup_hidden_concept(h, c, y, alpha=0.4, device=None):
    assert h.size()[0] == c.size()[0] == y.size()[0]
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(device)
    index = torch.randperm(h.size(0))
    mixed_h = lam * h + (1 - lam) * h[index, :]
    c_a, c_b = c, c[index, :]
    y_a, y_b = y, y[index]
    return mixed_h, c_a, c_b, y_a, y_b, lam


class MixupLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, output, target_a, target_b, lam):
        return lam * F.cross_entropy(output, target_a) + (1 - lam) * F.cross_entropy(output, target_b)
