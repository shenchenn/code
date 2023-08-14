import torch
import numpy as np
import os
import torchvision
# import cv2
from PIL import Image
from my_dataset import VOCSegmentation
from my_tool import get_transform
from pspnet.pspnet import PSPNet
from swfinet.shufflenet_single_scale import myModule
from swfinet.semseg import SemsegModel

backbone = myModule(pretrained=True,num_classes=11)
model = SemsegModel(backbone, 11)

message_model = torch.load('./save_weights/best_model_SwiftNet.pth',map_location=torch.device('cpu'))
epoch = message_model["epoch"]
missing_keys, unexpected_keys = model.load_state_dict(message_model["model"],)
if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    #加载测试模型参数情况
    print("加载测试模型参数情况")
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)
model.eval()

# image = Image.open('..data/camvid/data_dataset_voc/JPEGImages/0006R0_f01590.png').convert('RGB')
# label = Image.open('..data/camvid/data_dataset_voc/JPEGImages/0006R0_f01590.png')

# image = cv2.imread('C:/Users/GOFAesir/Desktop/imgtool/camvid/data_dataset_voc/JPEGImages/0001TP_006930.png')
# label = cv2.imread('C:/Users/GOFAesir/Desktop/imgtool/camvid/data_dataset_voc/SegmentationClassPNG/0001TP_006930.png')
# target = model_1(image)
# cv2.imshow("image",label)
dataset = VOCSegmentation('../data/camvid/',
                                    transforms=get_transform("test"),
                                    txt_name="test.txt")
loader = torch.utils.data.DataLoader(dataset ,
                                             batch_size=1,
                                             pin_memory=True,
                                             # collate_fn=val_dataset.collate_fn
                                             )
palette = [128, 128, 128,
           128, 0, 0,
           192, 192, 128,
           128, 64, 128,
           0, 0, 192,
           128, 128, 0,
           192, 128, 128,
           64, 64, 128,
           64, 0, 128,
           64, 64, 0,
           0, 128, 192]
i = 0
for img,lab in loader:
    i = i +1
    print(i)
    output = model(img)
    output_1 = torch.argmax(output,dim=1)
    output_1 = torch.squeeze(output_1,dim=0)
    output_1 = output_1.numpy().astype(np.uint8)
    im = Image.fromarray(output_1)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save("../SAGFM/outfile{}.png".format(i))

    img1 = Image.open("../SAGFM/outfile{}.png".format(i)).convert('P')
    img1.putpalette(palette)
    img1.save("../SAGFM/outfile{}.png".format(i))
    print("输出")
# cv2.waitKey(0) #等待按键

