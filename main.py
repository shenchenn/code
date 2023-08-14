import os
import torch
import numpy as np
import random
from torch.backends import cudnn


import argparse
from cr_model import create_model
from my_dataset import VOCSegmentation
from my_tool import get_transform
from my_tool import cal_fps
from train_and_eval import train_one_epoch
from train_and_eval import evaluate
from train_and_eval import test_saveModel

from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info
def main(args):
    #固定随机种子
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)

    # cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform("train"),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  transforms=get_transform("val"),
                                  txt_name="val.txt")

    test_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform("test"),
                                    txt_name="test.txt")

    # test_dataset = VOCSegmentation(args.data_path,
    #                               transforms=get_transform(train=False),
    #                               txt_name="test.txt")

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    num_workers = 8

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             # collate_fn=val_dataset.collate_fn
                                             )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               num_workers=num_workers,
                                               # shuffle=True,
                                               pin_memory=True,
                                               # collate_fn=train_dataset.collate_fn,
                                               # drop_last=True
                                              )

    model_name = "SwiftNet"
    run_code   = 'cal'
    model = create_model(num_classes=num_classes,model_name = model_name)


    model.to(device)

    fine_tune_factor = 4
    optim_params = [
        {'params': model.random_init_params(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': model.fine_tune_params(), 'lr': args.lr / fine_tune_factor,
         'weight_decay': args.weight_decay / fine_tune_factor},
    ]

    if run_code == 'train':
        scaler = torch.cuda.amp.GradScaler() if args.amp else None

        # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.99))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min)

        for epoch in range(0, args.epochs):
            loss_epoch = train_one_epoch(model, optimizer, args.lr,lr_scheduler, train_loader, device, epoch, args.epochs,scaler=scaler)
            print(loss_epoch)
            message,acc,miou = evaluate(model, val_loader, device, num_classes, epoch, args.epochs)
            print(str(message))
            #保存最优模型
            if epoch==0:
                # best_acc = acc
                best_miou = miou
            if miou > best_miou:
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_miou = miou
                model_file_name = './save_weights/best_model_{}.pth'.format(model_name)
                with open('./save_weights/best_model_{}.txt'.format(model_name), "a") as f:
                    epoch_bestmiou = f"[epoch: {epoch}][bestmiou: {best_miou}]\n"
                    f.write(epoch_bestmiou)
                    print(epoch_bestmiou)
                torch.save(state, model_file_name)

                # write into txt
            with open("./save_weights/message_{}.txt".format(model_name), "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {loss_epoch:.4f}\n"  \
                             # f"lr: {lr:.6f}\n"
                f.write(train_info + str(message) + "\n\n")
                # f.write(train_info + "\n\n")
            with open("./save_weights/loss_{}.txt".format(model_name), "a") as f:
                loss_s = f"{loss_epoch:.4f}\n"
                f.write(loss_s)

    elif run_code == 'test':
        print("进行测试集效果验证")
        message_test, Acc, Miou = test_saveModel(model,'./save_weights/best_model_{}.pth'.format(model_name), test_loader, device, num_classes)
        with open("./save_weights/test_message_{}.txt".format(model_name), "a") as f:
            f.write(str(message_test))
        print(str(message_test))

    elif run_code == 'cal':
        macs, param = get_model_complexity_info(model, (3, 720, 960), as_strings=True)
        print('macs:', macs, 'param:', param)
        cal_fps(model, device, test_loader)


def parse_args():

    parser = argparse.ArgumentParser(description="pytorch mycode_shenchen training")

    parser.add_argument("--data-path", default="../data/camvid/", help="自制数据集")
    parser.add_argument("--num-classes", default=11, type=int)                   #包括背景
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=12, type=int)
    parser.add_argument("--epochs", default=600, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=4e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr_min', default=1e-6, type=float, help='min learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
