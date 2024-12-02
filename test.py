import os

import torch
import sys


if __name__ == '__main__':
    print(sys.executable)  # 打印当前 Python 环境路径
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())