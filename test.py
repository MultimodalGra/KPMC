import os

import torch
import sys
import numpy as np


if __name__ == '__main__':
    file = np.load('./fs_cifar_16.npy',allow_pickle=True)
    print(type(file))
    print(file.ndim)
    print(file[0])
    # np.savetxt('cifar_npy_show.txt',file,fmt='%s')
