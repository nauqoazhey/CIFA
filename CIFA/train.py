import os
import numpy as np
import pandas as pd
from src.ILNNet import BINNet
from torch.utils.data import DataLoader
from src.DatasetClass import SimpleDataset, BalancedBatchSampler
from src.Datapreprocessing import load_wave_data, normalization, min_max_normal, HP_melsp
from src.Losses import CausalLoss
from src.options import args_parser
from tqdm import tqdm
from utils.AFP import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils.CMMD import cmmd_loss
from itertools import cycle



def expand_to_3_channels(x):
    if x.shape[1] == 2:
        extra = x[:, :1, :, :]
        x = torch.cat([x, extra], dim=1)
    return x

def update(minibatches, target_data, device, model, w_ca, cls_optimizer, dataclass, mmd_loss, epoch, num_classes, use_cmmd, use_mmd, cmmd_loss, mmd_start):
    correct_cls = 0.0
    total_cls = 0.0
    if dataclass in ['uav']:
        casual_loss = CausalLoss(3)

    src_x1, src_y1 = minibatches[0]
    src_x2, src_y2 = minibatches[1]
    cls_labels = torch.cat([src_y1, src_y2], dim=0).to(device)
    tgt_x, tgt_y = target_data
    tgt_x = tgt_x.float().to(device)
    src_x = torch.cat([src_x1, src_x2], dim=0)
    src_x = expand_to_3_channels(src_x)
    tgt_x = expand_to_3_channels(tgt_x)
    src_x = src_x.to(device).float()
    tgt_x = tgt_x.to(device).float()

    f1_map, f1_vec, logits, mmd_val, pseudo_labels, src_features, tgt_features = model(src_x, cls_labels, test_mode=False, src_x=src_x, tgt_x=tgt_x,
                                            src_y=cls_labels, mmd_loss=cmmd_loss, num_classes=num_classes, epoch=epoch, mmd_start=mmd_start)

    label_pred = torch.max(logits, dim=1)[1]
    total_cls += cls_labels.size(0)
    correct_cls += label_pred.eq(cls_labels).cpu().sum()

    fac_loss = casual_loss(f1_vec, cls_labels)
    cls_loss = F.cross_entropy(logits, cls_labels)

    lambda_mmd = min(0.5 * (epoch / args.ep), 0.5)

    if epoch >= mmd_start:
        if use_cmmd:
            mmd_loss_val = cmmd_loss(src_features, tgt_features, cls_labels, pseudo_labels, num_classes)
        elif use_mmd:
            mmd_loss_val = MMD_loss(src_features, tgt_features)
        else:
            mmd_loss_val = torch.tensor(0.0, device=cls_loss.device)
    else:
        mmd_loss_val = torch.tensor(0.0, device=cls_loss.device)
    total_loss = cls_loss + w_ca * fac_loss + lambda_mmd * mmd_loss_val
    cls_optimizer.zero_grad()
    total_loss.backward()
    cls_optimizer.step()

    loss_cls = cls_loss.detach().cpu().data.numpy()
    loss_fac = fac_loss.detach().cpu().data.numpy()
    losses = {'cls': loss_cls, 'causal_aggregation': loss_fac, 'mmd': mmd_loss_val, 'total': total_loss}
    acc_train = 100. * correct_cls / total_cls
    return losses, acc_train

def ReadData(domain, dataset):
    if dataset == '':
        if domain == '':
            file_path = ''
            audio_path = ''
        elif domain == '':
            file_path = ''
            audio_path = ''
        else:
            file_path = ''
            audio_path = ''

        file_name_label = pd.read_csv(file_path)
        data = file_name_label['']
        label = file_name_label['']

    data_list = []
    label_list = []

    for i, file_name in enumerate(data):
        audio_domain_data, sr = load_wave_data(audio_path, file_name)

        h_melsp, p_melsp = HP_melsp(audio_domain_data, sr, enhance_high=True)

        h_melsp = normalization(h_melsp)
        p_melsp = normalization(p_melsp)

        h_melsp = min_max_normal(h_melsp)
        p_melsp = min_max_normal(p_melsp)

        audio_domain_data = np.asarray([h_melsp, p_melsp])

        data_list.append(torch.tensor(audio_domain_data, dtype=torch.float32))
        label_list.append(label[i])

    data_by_label = {
        'data': data_list,
        'label': label_list,
    }
    return data_by_label

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def do_train(train_loaders_src, target_loader, model, device, cls_optimizer, epochs, out_dir, test_loader, cmmd_loss, mmd_loss, dataclass='uav'):
    best_train_accuracy = 0.0
    target_iter = cycle(target_loader)
    print("========= train begin =========")
    for epoch in tqdm(range(epochs)):
        model.train()
        total_correct = 0.0
        total_samples = 0.0
        batch_count = 0.0
        total_losses = {'cls': 0.0, 'causal_aggregation': 0.0, 'mmd': 0.0, 'total': 0.0}
        num_classes = 3 if dataclass in ['uav', 'uav_or'] else 4
        train_minibatches_iterator = zip(*[iter(loader) for loader in train_loaders_src])

        while True:
            try:
                minibatch_tuple = next(train_minibatches_iterator)
                if any(mb is None for mb in minibatch_tuple):
                    continue
                minibatches = list(minibatch_tuple)
                minibatch_tgt = next(target_iter)
                losses, acc_train = update(
                    minibatches=minibatches,
                    target_data=minibatch_tgt,
                    dataclass='uav',
                    device=device,
                    model=model,
                    w_ca=0.5,
                    cls_optimizer=cls_optimizer,
                    mmd_loss=MMD_loss,
                    cmmd_loss=cmmd_loss,
                    epoch=epoch,
                    num_classes=num_classes,
                    mmd_start=5,
                    use_cmmd=True,
                    use_mmd=False
                )
                for key in total_losses:
                    total_losses[key] += losses[key]

                total_correct += acc_train / 100.0 * len(minibatches[0][1])
                total_samples += len(minibatches[0][1])
                batch_count += 1

            except StopIteration:
                break

        if batch_count == 0:
            avg_losses = {key: 0.0 for key in total_losses}
            train_accuracy = 0.0
        else:
            avg_losses = {key: total_losses[key] / batch_count for key in total_losses}
            train_accuracy = (total_correct / total_samples) * 100.0

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_model_path = os.path.join(out_dir, ".pt")
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Loss_cls: {avg_losses['cls']:.4f}, "
              f"Loss_ca: {avg_losses['causal_aggregation']:.4f}, "
              f"Loss_mmd: {avg_losses['mmd']:.4f}, "
              f"Total_Loss: {avg_losses['total']:.4f}, "
              f"Train_Acc: {train_accuracy:.2f}%"
              f"(batch_count: {batch_count})")

        do_test(test_loader, model, device, best_model_path,epoch, train_loaders_src)

def do_test(test_loader,  model, device, best_model_path, epoch, train_loaders_src):
    print("========= test =========")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    total_correct = 0.0
    total_samples = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for minibatches in test_loader:
            data = minibatches[0].to(device)
            labels = minibatches[1].to(device)
            logits = model(data, labels, test_mode=True)[2]
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    test_accuracy = (total_correct / total_samples) * 100.0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # ========== matrices ==========
    cm = confusion_matrix(all_labels, all_preds)
    print("\n=== matrices ===")
    print(cm)
    return test_accuracy, precision, recall, f1

def main(idx, dataclass):
    epochs = args.ep
    batch_size = args.bs

    if dataclass in ['uav']:
        datasets_list = ['A', 'B', 'C']
    dataset_idx = list(range(len(datasets_list)))
    tgt_idx = [idx]
    src_idx = [i for i in dataset_idx if not tgt_idx.__contains__(i)]

    datasets_tgt = [datasets_list[i] for i in tgt_idx]
    datasets_src = [datasets_list[i] for i in src_idx]

    datasets_train_src = [ReadData(dataset, dataclass) for dataset in datasets_src]
    datasets_test_src = [ReadData(dataset, dataclass) for dataset in datasets_tgt]

    dataset_train1 = SimpleDataset(datasets_train_src[0])
    dataset_train2 = SimpleDataset(datasets_train_src[1])
    dataset_test = SimpleDataset(datasets_test_src[0])

    balanced_sampler1 = BalancedBatchSampler(dataset_train1.get_class_labels(), batch_size)
    balanced_sampler2 = BalancedBatchSampler(dataset_train2.get_class_labels(), batch_size)

    train_loader1 = DataLoader(dataset_train1, batch_sampler=balanced_sampler1)
    train_loader2 = DataLoader(dataset_train2, batch_sampler=balanced_sampler2)
    dataloader_params_test = dict(batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False)
    test_loader = DataLoader(dataset_test, **dataloader_params_test)
    train_loaders_src = [train_loader1, train_loader2]
    target_iter = iter(test_loader)

    if dataclass in ['uav', 'uav_or']:
        num_classes = 3
        model = BINNet(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    cls_optimizer = torch.optim.Adam(list(model.parameters()), lr=0.00005)
    mmd_loss = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)
    out_dir = "./output/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    do_train(train_loaders_src, target_iter, model, device, cls_optimizer, epochs, out_dir, test_loader, cmmd_loss, mmd_loss)


if __name__ == '__main__':
    args = args_parser()
    main(args.test, args.dataset)

