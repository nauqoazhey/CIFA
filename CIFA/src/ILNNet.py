import numpy as np
from econml.metalearners import XLearner
from sklearn.linear_model import LogisticRegression
from utils.SimpleLayerNorm import LayerNorm
from utils.SimpleLayerNorm import MixStyle
from utils.AFP import *


def generate_balanced_treatment(labels):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    treatment = np.zeros_like(labels, dtype=int)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) < 2:
            continue
        np.random.shuffle(label_indices)
        split_point = len(label_indices) // 2
        treatment[label_indices[:split_point]] = 1
    shuffled_indices = np.random.permutation(len(labels))
    return treatment[shuffled_indices]


class INBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(INBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class LNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = LayerNorm(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features=32, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2


class BINNet(nn.Module):
    def __init__(self, num_classes, lambda_uniform=0.1, inner_lr=1e-4, lambda_mmd=0.5):
        super(BINNet, self).__init__()

        self.afp = AFP()
        self.encoder_cla = FeatureEncoder()
        self.classifier = Classifier(num_classes)
        self.feature_weighting = None
        self.lambda_uniform = lambda_uniform
        self.inner_lr = inner_lr
        self.lambda_mmd = lambda_mmd
        self.mmd_loss = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=10)


    def forward(self, x, labels, test_mode=True, src_x=None, tgt_x=None, src_y=None, mmd_loss=None, cmmd_loss=None,num_classes=None,
                use_mmd=False, use_cmmd=True, mmd_start=5, epoch=0):
        mmd = torch.tensor(0.0, device=x.device)
        pseudo_labels = None
        src_features = torch.tensor([]).to(x.device)
        tgt_features = torch.tensor([]).to(x.device)
        h_x, p_x = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1)
        x_hp = self.afp(h_x, p_x)

        x1, flatten_x1 = self.encoder_cla(x_hp, test_mode=test_mode)

        if not test_mode:
            if self.feature_weighting is None:
                feature_dim = flatten_x1.size(1)
                causal_weights, topk_indices = self.causal_analysis(flatten_x1, labels)
                init_weights = causal_weights.mean(dim=0).detach()
                self.feature_weighting = FeatureWeightingBlock(feature_dim, init_causal_weights=init_weights).to(
                    flatten_x1.device)
                self.add_module("feature_weighting", self.feature_weighting)
                self.causal_idx = topk_indices

                x_weighted = flatten_x1 * causal_weights
            else:
                x_weighted = self.feature_weighting(flatten_x1)
        else:
            x_weighted = self.feature_weighting(flatten_x1) if self.feature_weighting is not None else flatten_x1

        pred_y = self.classifier(x_weighted)
        if (
                epoch >= mmd_start
                and not test_mode
                and tgt_x is not None
                and mmd_loss is not None
                and (use_cmmd or use_mmd)
        ):
            with torch.no_grad():
                tgt_h, tgt_p = tgt_x[:, 0, :, :].unsqueeze(1), tgt_x[:, 1, :, :].unsqueeze(1)
                tgt_hp = self.afp(tgt_h, tgt_p)
                _, tgt_flatten = self.encoder_cla(tgt_hp, test_mode=True)
                if self.feature_weighting:
                    tgt_flatten = self.feature_weighting(tgt_flatten)
                tgt_logits = self.classifier(tgt_flatten)
                tgt_probs = F.softmax(tgt_logits, dim=1)
                confidence, pseudo_labels_all = torch.max(tgt_probs, dim=1)
                mask = confidence > 0.8  # beta

                if mask.sum() < 4:  # alpha
                    use_cmmd_final = False
                    use_mmd_final = True
                    tgt_features = None
                    pseudo_labels = None
                else:
                    use_cmmd_final = True
                    use_mmd_final = False
                    pseudo_labels = pseudo_labels_all[mask]
                    tgt_features = tgt_flatten[mask]
            src_features = flatten_x1.detach()

            if tgt_features is not None:
                src_features = src_features.view(src_features.size(0), -1)
                tgt_features = tgt_features.view(tgt_features.size(0), -1)
            if self.feature_weighting is not None and self.causal_idx is not None and tgt_features is not None:
                src_features = self.get_causal_feature_subset(self.feature_weighting(src_features))
                tgt_features = self.get_causal_feature_subset(self.feature_weighting(tgt_features))
            else:
                src_features = self.feature_weighting(src_features) if self.feature_weighting else src_features
                tgt_features = self.feature_weighting(tgt_features) if self.feature_weighting else tgt_features

            if use_cmmd_final and cmmd_loss is not None and mask.any():
                if pseudo_labels is not None and src_y is not None and num_classes is not None:
                    mmd = cmmd_loss(src_features, tgt_features, src_y, pseudo_labels, num_classes)
                else:
                    mmd = MMD_loss(src_features, tgt_features)
            elif use_mmd_final:
                mmd = MMD_loss(src_features, tgt_features)
            else:
                mmd = torch.tensor(0.0, device=x.device)
        return x1, flatten_x1, pred_y, mmd, pseudo_labels, src_features, tgt_features

    def get_causal_feature_subset(self, features):

        if hasattr(self, "causal_idx") and self.causal_idx is not None:
            return features[:, self.causal_idx]
        else:
            print("[get_causal_feature_subset] Warning: causal_idx not set. Returning all features.")
            return features


    def causal_analysis(self, flatten_x1, cls_labels):

        flatten_x1_numpy = flatten_x1.detach().cpu().numpy()
        labels = cls_labels.detach().cpu().numpy()

        if flatten_x1_numpy.ndim != 2:
            raise ValueError(f"Expected flatten_x1_numpy to be 2D, but got shape {flatten_x1_numpy.shape}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels to be 1D, but got shape {labels.shape}")

        batch_size, num_features = flatten_x1_numpy.shape
        causal_weights = np.zeros((batch_size, num_features))
        topk_indices = []


        for i in range(num_features):
            single_feature = flatten_x1_numpy[:, i].reshape(-1, 1)

            treatment = generate_balanced_treatment(labels)
            if len(np.unique(labels)) < 2:
                print(f"Skipping feature {i}: only one class in labels.")
                continue

            # 使用X-Learner
            propensity_model = LogisticRegression()
            base_learner = LogisticRegression()
            x_learner = XLearner(models=base_learner, propensity_model=propensity_model)

            try:
                x_learner.fit(Y=labels, T=treatment, X=single_feature)
            except Exception as e:
                print(f"Error during x_learner.fit for feature {i}: {e}")
                continue

            # 计算因果效应
            try:
                causal_effect = x_learner.effect(single_feature, T0=0, T1=1)
                causal_weights[:, i] = causal_effect
            except Exception as e:
                print(f"Error during effect calculation for feature {i}: {e}")
                continue

            if np.abs(causal_effect).mean() > 0.1:
                topk_indices.append(i)

        return torch.tensor(causal_weights, dtype=torch.float32, device=flatten_x1.device), topk_indices


    def select_causal_features(self, causal_estimate):

        causal_weights = causal_estimate.value
        return torch.tensor(causal_weights)


class FeatureWeightingBlock(nn.Module):
    def __init__(self, num_features, init_causal_weights=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features))
        if init_causal_weights is not None:
            with torch.no_grad():
                self.weights.copy_(init_causal_weights)

    def forward(self, x):
        if x is None:
            return None
        weight = torch.sigmoid(self.weights).unsqueeze(0)
        weight_x = x + x * (1 + weight)
        return weight_x

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.mixstyle = MixStyle(p=0.5, alpha=0.3)
        self.in_block1 = INBlock(3, 16)
        self.in_block2 = INBlock(16, 32)
        self.ln_block1 = LNBlock(32, 64)
        self.ln_block4 = LNBlock(64, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x, test_mode=False):
        if not test_mode:
          x = self.mixstyle(x)
        x = self.in_block1(x)
        x = self.in_block2(x)
        x = self.ln_block1(x)
        x = self.dropout(x)
        x = self.ln_block4(x)
        x = self.pooling(x)
        flatten_x = self.flatten(x)
        return x, flatten_x
