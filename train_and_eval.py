import torch
import numpy as np
from tqdm import tqdm
from my_tool import adjust_learning_rate_poly

from torch import nn



class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


def evaluate(model, data_loader, device, num_classes,epoch,epochs):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    Loop = tqdm(data_loader)
    with torch.no_grad():
        for image, target in Loop:
            image, target = image.to(device), target.to(device)
            output = model(image)
            if type(output) is tuple:
                output = output[0]
            confmat.update(target.flatten(), output.argmax(1).flatten())
            acc_global, acc, iu = confmat.compute()

            acc = acc_global.item() * 100
            miou = iu.mean().item() * 100


            Loop.set_description(f'Eval [{epoch}/{epochs}]')
            Loop.set_postfix(acc_g = acc, miou = miou)

        acc_global, acc, iu = confmat.compute()
        acc_this_epoch = acc_global.item() * 100
        miou_this_epoch = iu.mean().item() * 100
        confmat.reduce_from_all_processes()
    return confmat , acc_this_epoch ,miou_this_epoch

def test_saveModel(model,model_path, data_loader, device, num_classes):
    message_model = torch.load(model_path)
    epoch = message_model["epoch"]
    missing_keys, unexpected_keys = model.load_state_dict(message_model["model"])
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        #加载测试模型参数情况
        print("加载测试模型参数情况")
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    Loop = tqdm(data_loader)
    with torch.no_grad():
        for image, target in Loop:
            image, target = image.to(device), target.to(device)
            output = model(image)
            if type(output) is tuple:
                output = output[0]
            confmat.update(target.flatten(), output.argmax(1).flatten())
            acc_global, acc, iu = confmat.compute()

            acc = acc_global.item() * 100
            miou = iu.mean().item() * 100


            Loop.set_description(f'Test [第{epoch}次迭代模型]')
            Loop.set_postfix(acc_g = acc, miou = miou)

        acc_global, acc, iu = confmat.compute()
        acc_this_epoch = acc_global.item() * 100
        miou_this_epoch = iu.mean().item() * 100
        confmat.reduce_from_all_processes()
    return confmat , acc_this_epoch ,miou_this_epoch


def train_one_epoch(model, optimizer, base_lr,lr_scheduler, data_loader, device, epoch,epochs,scaler = None):
    model.train()
    criterion = nn.CrossEntropyLoss( ignore_index=255, reduction='mean')
    loss_sum = 0
    loss_num = 0
    loop = tqdm(data_loader)
    for image, target in loop:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            loss = 0
            if type(output) is tuple:   # output = (main_loss, aux_loss1, axu_loss2***)
                length = len(output)
                for index, out in enumerate(output):
                    loss_record = criterion(out, target)
                    if index == 0:
                        loss_record *= 0.6
                    else:
                        loss_record *= 0.4 / (length - 1)
                    loss += loss_record
            else:
                loss = criterion(output, target)
            # loss1 = nn.functional.cross_entropy(output, target, ignore_index=255)
            # loss2 = nn.functional.cross_entropy(aux, target, ignore_index=255)
            # loss = loss1 + 0.4 * loss2
        loss_sum = loss_sum + loss.item()
        loss_num = loss_num + 1
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        ################学习率调整
        # lr_adjust = adjust_learning_rate_poly(optimizer = optimizer, epoch = epoch, num_epochs = epochs, base_lr = base_lr, power = 0.9)
        lr_scheduler.step()
        loop.set_description(f'Epoch [{epoch}/{epochs}]')
        loop.set_postfix(loss=loss_sum/loss_num)
    return loss_sum/loss_num