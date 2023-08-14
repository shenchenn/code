#calculate the mean and std for dataset
#The mean and std will be used in src/lib/datasets/dataset/oxfordhand.py line17-20
#The size of images in dataset must be the same, if it is not same, we can use reshape_images.py to change the size

import os
# from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
import cv2

filepath = 'C:/Users/GOFAesir/Desktop/imgtool/data/data_dataset_voc/JPEGImages'  # 图片目录
train_name_list = []         #训练集图片名称
with open("C:/Users/GOFAesir/Desktop/imgtool/data/data_dataset_voc/imgset/train.txt") as f:
    for line in f.readlines():
        temp = line.split()
        train_name_list.extend(temp)

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(train_name_list)):
    filename = train_name_list[idx]
    img = cv2.imread(os.path.join(filepath, filename + ".jpg" )) / 255.0
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(train_name_list) * 1280 * 720  # 这里是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(train_name_list)):
    filename = train_name_list[idx]
    img = cv2.imread(os.path.join(filepath, filename + ".jpg")) / 255.0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
#7 1.5 1.5
# R_mean is 0.446415, G_mean is 0.456023, B_mean is 0.466357
# R_var is 0.237943, G_var is 0.226932, B_var is 0.223950
#6 2 2
# R_mean is 0.446017, G_mean is 0.455584, B_mean is 0.465820
# R_var is 0.238472, G_var is 0.227074, B_var is
#7 1 2

# R_mean is 0.446415, G_mean is 0.456023, B_mean is 0.466357
# R_var is 0.237943, G_var is 0.226932, B_var is 0.223950

#camvid
