# 该脚本基于 commit bc424305 所得 IO trace 开发

import json
import time
import numpy as np
import torch

def delayMicrosecond(t):    # 微秒级延时函数
    start,end=0,0           # 声明变量
    start=time.time()       # 记录开始时间
    t=(t-0)/1000000     # 将输入t的单位转换为秒，-x是时间补偿
    while end-start<t:  # 循环至时间差值大于或等于设定值时
        end=time.time()     # 记录结束时间


# KVcache_data具体大小要根据模型参数手动指定,否则assert
Max_KVcache_size=[543,32,64]
# KVcache_data=torch.ones(Max_KVcache_size,dtype=torch.float16,device="cpu")
# KVcache_data=torch.zeros([4096,32,64],dtype=torch.float16,device="cpu")
# temp_data=KVcache_data[:100,:,:] #预热

# kvpath="./"
kvpath="/mnt/md0/" #需要先挂载RAID至该目录

# jsonpath="../IOtrace.json"
jsonpath="./all_dram.json"

stored_kcache_num=0
stored_vcache_num=0

with open(jsonpath,'r',encoding='utf8')as fp:
    json_datas = json.load(fp)
    all_len=len(json_datas)
    
    for json_data in json_datas[:1000]:
        if len(json_data) ==1 and str(json_data)[2:9] == "session":
            print("get session")
            stored_kcache_num=0
            stored_vcache_num=0
            # todo：若后续实现打断重启的多轮对话，此处应改变stored_kcache_num，和stored_vcache_num
            # todo
        elif len(json_data) ==1 and str(json_data)[2:3] == "q":
            print("get q")
            # todo：若后续实现打断重启的多轮对话，此处应改变stored_kcache_num，和stored_vcache_num
        else:
            if json_data["opration"] == "load weight":
            # 当前weight默认都在GPU中，不做卸载
                # if json_data["dtype"] == "torch.float16":
                #     data_size=int(json_data["size"]*2)
                # else:
                #     print("【WARNING】data_size unknown")
                pass
            elif json_data["opration"] == "compute":
                sleep_time = int(float(json_data["time_cost(s)"][0])*1000000)
                time_start=time.time() 
                delayMicrosecond(sleep_time)
                time_end=time.time() 
                print("\n  compute_time="+str(sleep_time)+" -- "+str(int((time_end-time_start)*1000000)))

            elif json_data["opration"] == "store kcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                print("")

                token_len=int(json_data["session_len"])
                time_start=time.time() 
                data=torch.ones([token_len,Max_KVcache_size[1],Max_KVcache_size[2]],dtype=torch.float16,device="cpu")
                data_malloc_time=time.time() - time_start
                print("  kdata_malloc_time="+str(int(data_malloc_time*1000000)))

                time_start=time.time() 
                torch.save(data, kvpath+"kcache.pt")
                data_save_time=time.time() - time_start
                print("  kdata_save_time="+str(int(data_save_time*1000000)))

            elif json_data["opration"] == "store vcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                print("")

                token_len=int(json_data["session_len"])
                time_start=time.time() 
                data=torch.ones([token_len,Max_KVcache_size[1],Max_KVcache_size[2]],dtype=torch.float16,device="cpu")
                data_malloc_time=time.time() - time_start
                print("  vdata_malloc_time="+str(int(data_malloc_time*1000000)))

                time_start=time.time() 
                torch.save(data, kvpath+"vcache.pt")
                data_save_time=time.time() - time_start
                print("  vdata_save_time="+str(int(data_save_time*1000000)))

            elif json_data["opration"] == "load kcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                print("")

                time_start=time.time() 
                data = torch.load(kvpath+"kcache.pt", map_location=lambda storage, loc: storage,weights_only=True)
                data_save_time=time.time() - time_start
                print("  kdata_load_time="+str(int(data_save_time*1000000)))


            elif json_data["opration"] == "load vcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                print("")

                time_start=time.time() 
                data = torch.load(kvpath+"vcache.pt", map_location=lambda storage, loc: storage,weights_only=True)
                data_save_time=time.time() - time_start
                print("  vdata_load_time="+str(int(data_save_time*1000000)))

            else:
                assert 0
                pass
        

# todo：save只需增量，load是全量
# todo: 没有模拟出flexgen的IO方式
# todo: 没有模拟多GPU处理场景
# todo: 没有模拟出多有限KVcache预算下，KVcahe删减的场景
# todo：暂不支持打断重启对话