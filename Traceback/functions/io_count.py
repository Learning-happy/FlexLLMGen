import subprocess
import time
import re


def iostat_size(device:str):
    iostat_output = subprocess.check_output(["iostat", device]).decode('utf-8')
    # 使用正则表达式解析iostat输出
    stats = {
        'kB_read': -1,
        'kB_wrtn': -1
    }
    lines = iostat_output.strip().split('\n')
    for line in lines[2:]:  # 跳过前两行标题
        columns = line.split()
        # print("columns=",columns)
        if len(columns) == 8:
            if columns[0] == "Device":
                assert columns[5] == "kB_read"
                assert columns[6] == "kB_wrtn"
            if columns[0] == device:
                stats['kB_read'] = int(float(columns[5]))  # 读取数据量
                stats['kB_wrtn'] = int(float(columns[6]))  # 写入数据量
                break
    if stats['kB_read']==-1 or stats['kB_wrtn']==-1:
        print("Warning fail to get iostat, device=",device,"\n",iostat_output)
        assert 0
    return stats


# 注意：show_num取1时，通常不准确，
def iostat_speed(show_num:int, device:str):
    if show_num == 1:
        passiostat_output = subprocess.check_output(["iostat","-x",device]).decode('utf-8')
    else:
        iostat_output = subprocess.check_output(["iostat","-x","1",str(show_num),device]).decode('utf-8')
    # 使用正则表达式解析iostat输出
    stats = {
        'rkB/s': 0,
        'wkB/s': 0
    }
    lines = iostat_output.strip().split('\n')
    get_device = False
    for line in lines[2:]:  # 跳过前两行标题
        columns = line.split()
        # print("columns=",columns)
        if len(columns) == 21:
            if columns[0] == "Device":
                assert columns[2] == "rkB/s"
                assert columns[8] == "wkB/s"
            if columns[0] == device:
                get_device = True
                # print("columns=",columns)
                stats['rkB/s'] += int(float(columns[2]))  # 读取数据量
                stats['wkB/s'] += int(float(columns[8]))  # 写入数据量
    if not get_device:
        print("Warning fail to get iostat, device=",device,"\n",iostat_output)
        assert 0
    stats['rkB/s'] = stats['rkB/s']/show_num
    stats['wkB/s'] = stats['wkB/s']/show_num

    return stats

# 设备名称
# device = "sda2"
device = "md0"
# device = "nvme4n1"

st = time.time()
initial_iostat_size = iostat_size(device)
print("[initial_iostat_size]=",time.time()-st)
time.sleep(10)
st = time.time()
final_iostat_size = iostat_size(device)
print("[final_iostat_size]=",time.time()-st)
st = time.time()
mean_iostat_speed = iostat_speed(11,device)
print("[mean_iostat_speed]=",time.time()-st)

MB_read = (final_iostat_size['kB_read'] - initial_iostat_size['kB_read'])/1024
MB_wrtn = (final_iostat_size['kB_wrtn'] - initial_iostat_size['kB_wrtn'])/1024


print(f"Total read  IO in 10s: {int(MB_read)} MB, {int(mean_iostat_speed['rkB/s']/1024)} MB/s")
print(f"Total write IO in 10s: {int(MB_wrtn)} MB, {int(mean_iostat_speed['wkB/s']/1024)} MB/s")

