import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, flatten_features, data_label):
        list_category = [[] for i in range(self.num_classes)]
        for i, fv in zip(data_label, flatten_features):
            list_category[i].append(fv.unsqueeze(0))
            total_causal_loss = 0
            valid_class_count = 0
        for i in range(self.num_classes):
            if len(list_category[i]) == 0:
                continue
            fm_i = torch.cat(tuple(list_category[i]), dim=0)
            # print(fm_i.shape)
            fm_i_mean = torch.mean(fm_i, dim=0, keepdim=True)
            causal_i = torch.sum(torch.mean((fm_i - fm_i_mean).pow(2), dim=0))
            total_causal_loss = total_causal_loss + causal_i

        return total_causal_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.1 * off_diag

    return loss

