import torch
import mmap
import os
import numpy as np
import ctypes

HEAD=32
DEM=32

# 创建一个初始的全1的tensor
oneCache = torch.ones([10, HEAD, DEM], dtype=torch.float16, device="cpu")

# 指定文件名
filename = 'oneCache.pt'

# 确保文件存在，并且大小足够
file_size = oneCache.numel() * oneCache.dtype.itemsize
with open(filename, 'wb') as f:
    f.truncate(file_size)

# 打开文件并创建内存映射
with open(filename, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

    # 将tensor的数据写入内存映射文件
    buffer = oneCache.cpu().numpy().tobytes()  # 将tensor转换为bytes
    mm.write(buffer)
    mm.flush()  # 确保数据写入磁盘

# 定义一个函数来更新内存映射文件中的特定切片
def update_slice(tensor, index):
    with open(filename, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        # 计算切片的起始位置
        start = index * HEAD * DEM * tensor.dtype.itemsize
        # 将切片数据写入内存映射文件
        buffer = tensor.cpu().numpy().tobytes()  # 将切片tensor转换为bytes
        mm.seek(start)  # 移动到指定的起始位置
        mm.write(buffer)
        mm.flush()  # 确保数据写入磁盘
        mm.close()

# 定义一个函数来读取内存映射文件中的特定切片
def read_slice(index):
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # 计算切片的起始位置
        start = index * HEAD * DEM * oneCache.dtype.itemsize
        # 读取切片数据
        buffer = mm.read(HEAD * DEM * oneCache.dtype.itemsize)
        mm.close()
        # 将bytes转换为numpy数组并返回
        return np.frombuffer(buffer, dtype=np.float16).reshape(HEAD, DEM)

# 模拟tensor的[1,:,:]切片部分随机变化
for _ in range(500000):  # 假设我们更新5次
    updated_slice = torch.randn(HEAD, DEM, dtype=torch.float16)
    update_slice(updated_slice, 1)  # 更新文件中的第2个切片（索引从0开始）
    print("Updated slice at index 1:")
    print(read_slice(1))

# 清理资源
if mm:
    mm.close()