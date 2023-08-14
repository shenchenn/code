import os
import numpy as np

f_train = open('./data_dataset_voc/imgset/train.txt', 'w')
f_val   = open('./data_dataset_voc/imgset/val.txt', 'w')
p='./data_dataset_voc/JPEGImages'

filenames=os.listdir(p)
np.random.seed(42)
train_ratio = 0.8
shuffled_indices = np.random.permutation(len(filenames))
train_set_size = int(len(filenames)) * train_ratio
train_indices = shuffled_indices[:train_set_size]
val_indices = shuffled_indices[train_set_size:]

filenames_train = filenames[train_indices]
filenames_val   = filenames[val_indices]

for filename in filenames_train:
    filename1=os.path.splitext(filename)
    print(filename1[0])
    f_train.write(filename1[0]+'\n')
f_train.close()

for filename in filenames_val:
    filename1=os.path.splitext(filename)
    print(filename1[0])
    f_val.write(filename1[0]+'\n')
f_val.close()
