import json
import time
import numpy as np
import torch
import random

Max_KVcache_size=[543,32,64] # KVcache_data具体大小要根据模型参数手动指定来通过检查,否则assert
oneCache=torch.ones([1,32,64],dtype=torch.float16,device="cpu")
MaxAttLayerId=48

kvpath="./kvcache/"
# kvpath="/mnt/md0/" #需要先挂载RAID至该目录
json_metapath="../Tracefile/q_num_for_every_session.json"
json_path="../Tracefile/"

def timePrint(strs):
    print(strs) # 注释此行将不打印每步的延迟
    pass

def delayMicrosecond(t):    # 微秒级延时函数
    start,end=0,0           # 声明变量
    start=time.time()       # 记录开始时间
    t=(t-0)/1000000     # 将输入t的单位转换为秒，-x是时间补偿
    while end-start<t:  # 循环至时间差值大于或等于设定值时
        end=time.time()     # 记录结束时间

finish_time=0
R_time=0
W_time=0


session_nums=0 #共有几个session
session_left_qnums=[] # 每个session还剩几个q没处理
session_active=[] # 还没处理完的session的id集合
session_stored_tokens=[] # 已存储完KVcache的token的id集合

with open(json_metapath,'r',encoding='utf8')as fp:
    json_datas = json.load(fp)
    session_nums=len(json_datas)
    session_index=0
    for _ , qnum in json_datas.items():
        session_left_qnums.append(int(qnum))
        session_active.append(session_index)
        session_stored_tokens.append(0)
        session_index+=1
print(session_left_qnums)
print(session_active)

while len(session_active) != 0:
    session_active_index = random.randint(0,len(session_active)-1)
    session_id = session_active[session_active_index]
    print("session_id="+str(session_id)+"    session_left_qnums="+str(session_left_qnums))
    filepath=json_path+"session "+str(session_id)+" trace.json"

    with open(filepath,'r',encoding='utf8')as fp:
        json_datas = json.load(fp)
        num_q=int(json_datas[0]["num_q"])
        qid = num_q-session_left_qnums[session_id] + 1 
        assert qid==json_datas[qid]["q"]
        qinfo=json_datas[qid]["I/O info"]
        
        for json_data in qinfo:
            nameHead="S"+str(session_id)+"L"+str(json_data["layer_id"])

            if json_data["opration"] == "load weight":
            # 当前weight默认都在GPU中，不做卸载
                # if json_data["dtype"] == "torch.float16":
                #     data_size=int(json_data["size"]*2)
                # else:
                #     print("【WARNING】data_size unknown")
                pass

            elif json_data["opration"] == "compute":
                sleep_time = int(float(json_data["time_cost(s)"])*1000000)
                time_start=time.time() 
                delayMicrosecond(sleep_time)
                time_interval=int((time.time() - time_start)*1000000)
                # timePrint("\n  compute_time="+str(sleep_time)+" -- "+str(time_interval))
                finish_time+=time_interval

            elif json_data["opration"] == "store kcache" or json_data["opration"] == "store vcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                token_len=int(json_data["session_len"])
                for i in range(token_len-session_stored_tokens[session_id]):
                    token_id = i+1+session_stored_tokens[session_id]
                    pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                    time_start=time.time()
                    torch.save(oneCache, pt_name)
                    time_interval=int((time.time() - time_start)*1000000)
                    # timePrint("  "+str(json_data["opration"][-6:-5])+"_saveTime="+str(time_interval))
                    finish_time+=time_interval
                    W_time+=time_interval
                if(json_data["layer_id"] == MaxAttLayerId): # store增量，#todo：异步IO适配
                    session_stored_tokens[session_id]=token_len

            elif json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache":
                assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                token_len=int(json_data["session_len"])
                for i in range(token_len-1):
                    token_id = i+1
                    pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                    time_start=time.time() 
                    data = torch.load(pt_name, map_location=lambda storage, loc: storage,weights_only=True)
                    time_interval=int((time.time() - time_start)*1000000)
                    # timePrint("  "+str(json_data["opration"][-6:-5])+"_loadTime="+str(time_interval))
                    finish_time+=time_interval
                    R_time+=time_interval

            elif json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                assert str(Max_KVcache_size)[1:-1] == str(json_data["shape"][1:-1])
                token_len=int(json_data["history_len"])
                for i in range(token_len):
                    token_id = i+1
                    pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                    time_start=time.time() 
                    data = torch.load(pt_name, map_location=lambda storage, loc: storage,weights_only=True)
                    time_interval=int((time.time() - time_start)*1000000)
                    # timePrint("  his_"+str(json_data["opration"][-6:-5])+"_load_time="+str(time_interval))
                    finish_time+=time_interval
                    R_time+=time_interval

            else:
                print("WARNING"+str(session_id)+"-"+str(qid)+":\n"+str(json_data))
                assert 0


    session_left_qnums[session_id]-=1
    if(session_left_qnums[session_id]==0):
        session_active.pop(session_active_index)
        print("session_active changed:" + str(session_active) + "    deactivate:"+str(session_id))


print("ALL_time:"+str(finish_time/1000)+" ms")
print("IO_time:"+str((R_time+W_time)/1000)+"ms")
print("R_time:"+str((R_time)/1000)+"ms")
print("W_time:"+str((W_time)/1000)+"ms")

# # todo: 没有模拟出flexgen的IO方式（异步IO）
# # todo: 没有模拟多GPU处理场景
# # todo: 没有模拟出多有限KVcache预算下，KVcahe删减的场景
# # todo：没有考虑tensor从内存拷贝到GPU的时间