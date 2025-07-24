import torch
import torch.nn as nn
import torch.nn.functional as F


class AFP(nn.Module):
    def __init__(self):
        super(AFP, self).__init__()

        self.branch1_1 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Conv2d(3, 3, 3, padding=1),
        )
        self.branch_SE1 = SE_block(channel=3)

        self.branch2_1 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.Conv2d(3, 3, 3, padding=1)
        )
        self.branch_SE2 = SE_block(channel=3)

        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        b1_1 = self.branch1_1(x)
        b1 = self.branch_SE1(b1_1)

        b2_1 = self.branch2_1(y)
        b2 = self.branch_SE2(b2_1)

        w1 = torch.exp(self.w) / torch.sum(torch.exp(self.w))

        x_out = b1 * w1 + b2 * (1 - w1)
        return x_out

class SE_block(nn.Module):
    def __init__(self, channel, r=0.5):
        super(SE_block, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)
        weight = self.fc(branch)

        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        scale = weight * x
        return scale


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.mean(L2_distance.data[L2_distance.data > 0])
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if source is None or target is None:
            device = source.device if source is not None else 'cpu'
            return torch.tensor(0.0, device=device)

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.gaussian_kernel(source, target)
            XX = kernels[:batch_size, :batch_size].mean()
            YY = kernels[batch_size:, batch_size:].mean()
            XY = kernels[:batch_size, batch_size:].mean()
            YX = kernels[batch_size:, :batch_size].mean()
            loss = XX + YY - XY - YX
        else:
            raise ValueError(f"Unknown kernel_type {self.kernel_type}")
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=source.device)
        return loss