import torch
import torch.nn.functional as F

def mmd_rbf(x, y, gamma=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (x ** 2).sum(dim=1).unsqueeze(1)
    ry = (y ** 2).sum(dim=1).unsqueeze(0)

    dist_xx = rx + rx.t() - 2 * xx
    dist_yy = ry + ry.t() - 2 * yy
    dist_xy = rx + ry - 2 * xy

    Kxx = torch.exp(-gamma * dist_xx)
    Kyy = torch.exp(-gamma * dist_yy)
    Kxy = torch.exp(-gamma * dist_xy)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def cmmd_loss(src_features, tgt_features, src_labels, tgt_labels, num_classes, gamma=1.0):

    if src_features is None or tgt_features is None:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    if src_labels is None or tgt_labels is None:
        return torch.tensor(0.0, device=src_features.device)
    if not isinstance(src_labels, torch.Tensor):
        src_labels = torch.tensor(src_labels, device=src_features.device)
    if not isinstance(tgt_labels, torch.Tensor):
        tgt_labels = torch.tensor(tgt_labels, device=tgt_features.device)

    loss = 0.0
    count = 0

    for c in range(num_classes):
        src_idx = (src_labels == c)
        tgt_idx = (tgt_labels == c)

        if src_idx.sum().item() == 0 or tgt_idx.sum().item() == 0:
            continue

        src_c = src_features[src_idx]
        tgt_c = tgt_features[tgt_idx]
        loss += mmd_rbf(src_c, tgt_c, gamma)
        count += 1

    if count > 0:
        loss /= count
    else:
        loss = torch.tensor(0.0, device=src_features.device)

    return loss
