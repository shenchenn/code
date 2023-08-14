import transforms as T
import torch
import numpy as np
from tqdm import tqdm
import more_itertools

## 图像预处理
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.434293, 0.426768, 0.413553), std=(0.310211, 0.314617, 0.310715)):
        min_size = int(0.5 * base_size)
        max_size = int(2 * base_size)
        # trans = [T.RandomResize(base_size)]
        trans = [T.RandomResize(min_size, max_size)]
        trans.extend([
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self,  mean=(0.434293, 0.426768, 0.413553), std=(0.310211, 0.314617, 0.310715)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetTest:
    def __init__(self,  mean=(0.434293, 0.426768, 0.413553), std=(0.310211, 0.314617, 0.310715)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(dataSetname):
    if dataSetname=="train":
        base_size = 720
        crop_size = 448
        hflip_prob = 0.5
        return SegmentationPresetTrain(base_size, crop_size ,hflip_prob)
    elif dataSetname=="val":
        return SegmentationPresetEval()
    elif dataSetname=="test":
        return SegmentationPresetTest()

def cal_fps(model, device, test_loader):
    ci = 0
    for i in range(3):
        len = more_itertools.ilen(test_loader)
        model.eval()
        model.to(device)
        # 预热，防止GPU处于休眠状态
        random_input = torch.rand(1, 3, 720, 960).to(device)
        print("GPU预热...")
        with torch.no_grad():
            for _ in range(1500):
                _ = model(random_input)

        # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
        torch.cuda.synchronize()
        # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        timings = np.zeros((len, 1))

        print('FPS计算中...\n')
        rep = 0

        with torch.no_grad():
            loop = tqdm(test_loader)
            for image, target in loop:
                image, target = image.to(device), target.to(device)
                starter.record()
                _ = model(image)
                ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
                timings[rep] = curr_time
                rep = rep + 1

        avg = timings.sum() / len
        print('\nfps={}\n'.format(1000/avg))
        with open("./save_weights/fps.txt", "a") as f:
            fps = f"{1000/avg:.4f}\n"
            f.write(fps)
        ci = ci + 1000/avg
    with open("./save_weights/fps.txt", "a") as f:
        fps = f"{ci/3:.4f}\n"
        f.write(fps)



def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr






