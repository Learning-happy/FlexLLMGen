import json
import time
import numpy as np
import torch
import random
from multiprocessing import Process,Pool,Pipe,Queue,Manager
import mmap
import os
import ctypes

# 目标GPU与原GPU的推理性能之比（原GPU：录制trace文件的GPU），默认为 1
GPU_SPEED_UP = 1 
# [不推荐使用] 目标LLM与原LLM的大小之比（原LLM：录制trace文件的LLM），默认为 1
# 注意:模型层数无法改变，仅scale了每层的参数量
LLM_SIZE_SCALING = 1 

GPU_NUM = 2 # 模拟的GPU（LLM实例）数量
PCIE_BW = 16 # PCIe带宽，用于计算CPU与GPU之间的数据搬运开销，单位：GB/S
MAX_Q = -1 # Trace回放数量，以Q为单位：考虑到测试集太大，可以推理到一定的数量就停止回放；默认值-1，即全部测试完

HEAD_NUM = 32 #注意力头数，需根据模型手动指定，当前为 opt1.3B
WEIGHT_DEM = 64*LLM_SIZE_SCALING #权重维度，需根据模型手动指定，当前为 opt1.3B
MAX_ATT_LAYER_ID=48 # attention的最后一层在全局的 Layer Id，需根据模型手动指定，当前为 opt1.3B
ATT_LAYER_NUM=24 # attention的最后一层在全局的 Layer Id，需根据模型手动指定，当前为 opt1.3B
oneCache=torch.ones([1,1,HEAD_NUM,WEIGHT_DEM],dtype=torch.float16,device="cpu")
MAX_TOKENS = 512 #最大KVcahe预算，当前实现了滑动窗口法
Max_KVcache=torch.ones([ATT_LAYER_NUM,MAX_TOKENS,HEAD_NUM,WEIGHT_DEM],dtype=torch.float16,device="cpu")

KVPATH = "./kvcache/" # 存放KVcache的目录
# KVPATH = "/mnt/md0/kvcache/" #需要先挂载 RAID 至/mnt/md0/
# KVPATH = "/mnt/ssd/kvcache/" #需要先挂载 femu盘 至/mnt/ssd/
json_path = "../Tracefile/"
# json_path = "/home/femu/MoreTracefile/101sessions_opt1-3/Tracefile/"
json_metapath = json_path + "q_num_for_every_session.json"

def delayMicrosecond(t):    # 微秒级延时函数
    start,end=0,0           # 声明变量
    start=time.time()       # 记录开始时间
    t=(t-0)/1000000     # 将输入t的单位转换为秒，-x是时间补偿
    while end-start<t:  # 循环至时间差值大于或等于设定值时
        end=time.time()     # 记录结束时间

def del_ptfile(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path) and str(file_path)[-3:]==".pt":
            os.remove(file_path)

# 定义一个函数来更新内存映射文件中的特定切片
def update_mmapfile(mm, all_index:list,layer_index):
    stime = time.time()
    att_layer_id = int(layer_index/2-1)
    updated_slice = torch.randn(1,HEAD_NUM, WEIGHT_DEM, dtype=torch.float16)
    buffer = updated_slice.cpu().numpy().tobytes()  # 将切片tensor转换为bytes
    assert mm!=None
    for index in all_index:
        # 计算切片的起始位置
        # start = index * HEAD_NUM * WEIGHT_DEM * updated_slice.dtype.itemsize
        start = int(att_layer_id*MAX_TOKENS*HEAD_NUM*WEIGHT_DEM + index*HEAD_NUM*WEIGHT_DEM)
                # 将切片数据写入内存映射文件
        mm.seek(start)  # 移动到指定的起始位置
        mm.write(buffer)
    # mm.flush()  # 确保数据写入磁盘
    return int((time.time() - stime)*1000000)

# 定义一个函数来读取内存映射文件中的特定切片
def read_mmapfile(mm, max_index,layer_index):
    stime = time.time()
    att_layer_id = int(layer_index/2-1)
    assert mm!=None
    # 计算切片的起始位置
    mm.seek(att_layer_id*MAX_TOKENS*HEAD_NUM*WEIGHT_DEM)
    buffer = mm.read(max_index*HEAD_NUM*WEIGHT_DEM * oneCache.dtype.itemsize)
    return [int((time.time() - stime)*1000000), buffer]

def aio_trace(io_submit_queue:Queue, io_finish_queue:Queue):
    saveCache = oneCache
    listening1=True
    session_id = -1
    fp = None
    mm = None
    while True: # 持续处理，直到父进程通知其结束
        listening2=True
        while(listening2): # 持续接听，直到获取新任务
            if not io_submit_queue.empty():
                recv = io_submit_queue.get()
                if len(recv) == 1:
                    assert recv[0] == -1
                    listening1=False
                listening2=False
            else:
                delayMicrosecond(100)

        if listening1 == False:
            if mm != None:
                mm.flush()
                mm.close()
            if fp != None:
                fp.close()
            break
        
        # print("do io :",recv)
        IO_wait_time=int((time.time() - recv[6])*1000000)
        filename = KVPATH+"S"+str(recv[3])+".pt"
        # IO submit 格式 1：["R", "k"或者"v", Gen_token_id, session_id, layer_id, token_len, io提交时间]
        # IO submit 格式 2：["W", "k"或者"v", Gen_token_id, session_id, layer_id, token_ids:list, io提交时间, creat?:bool]
        # IO submit 格式 3：["F", -1,-1,-1,-1,-1, io提交时间]
        time_start=time.time()

        # fp和 mm检查
        if recv[0] == "R" or (recv[0] == "W" and (session_id!= recv[3] or recv[7])):
            if not os.path.exists(filename):
                if recv[0] == "R":
                    assert 0
                if recv[0] == "W":
                    assert mm == None
                    assert fp == None
                    fp = open(filename, 'wb')
                    buffer_size = Max_KVcache.numel() * Max_KVcache.dtype.itemsize
                    fp.truncate(buffer_size)
                    fp.close()
                    fp=None
            if fp == None and mm == None:
                fp = open(filename, 'r+b')
                mm = mmap.mmap(fp.fileno(), 0)
            elif fp != None and mm != None:
                pass
            else:
                assert 0

        # 执行 IO
        if recv[0] == "R":
            read_mmapfile(mm, recv[5], recv[4])
            copy_latency = recv[5] * (saveCache.numel() * saveCache.element_size()) / (PCIE_BW * 1000000000) * 1000000 # 以微妙为单位
        if recv[0] == "W":
            update_mmapfile(mm, recv[5],recv[4])
            copy_latency = len(recv[5]) * (saveCache.numel() * saveCache.element_size()) / (PCIE_BW * 1000000000) * 1000000 # 以微妙为单位
        if recv[0] == "F":
            assert mm != None
            mm.flush()
            mm.close()
            mm=None
            assert fp != None
            fp.close()
            fp=None
            copy_latency=0
        delayMicrosecond(copy_latency)
        RW_time_interval=int((time.time() - time_start)*1000000)

        # 返回结果
        # IO finish 格式：["R"或者"W", "k"或者"v", Gen_token_id, layer_id, 文件读写时间，cp时间, IO等待时间） ]
        # 或者 ["F", FLUSH时间, IO等待时间]
        if recv[0] == "F":
            io_finish_queue.put(["F",RW_time_interval,IO_wait_time])
        else:
            io_finish_queue.put([recv[0],recv[1],recv[2],recv[4],RW_time_interval-copy_latency,copy_latency,IO_wait_time])

def check_io(io_finish_queue:Queue,io_submiting:list,IO_wait_time:int,R_time:int,W_time:int,CP_time:int):
    tp_IO_wait_time = IO_wait_time
    tp_R_time = R_time
    tp_W_time = W_time
    tp_cp_time = CP_time
    while not io_finish_queue.empty():
        # IO finish 格式：["R"或者"W", "k"或者"v", Gen_token_id, layer_id, 文件读写时间，cp时间, IO等待时间） ]
        # 或者 ["F", FLUSH时间, IO等待时间]
        recv = io_finish_queue.get()
        if recv[0] == "F":
            tp_W_time += recv[1]
            tp_IO_wait_time += recv[2]
            # print("flush time=",recv[1])
            assert io_submiting[0] == "F"
            io_submiting.pop(0) 
            continue
        index = 0
        for item in io_submiting:
            if item[0] == recv[0] and item[1] == recv[1] and item[2] == recv[2] and item[3] == recv[3]:
                io_submiting.pop(index) # 该IO结束，可以从异步等待队列移除
                if recv[0] == "R":
                    tp_R_time += recv[4]
                if recv[0] == "W":
                    tp_W_time += recv[4]
                tp_cp_time += recv[5]
                tp_IO_wait_time += recv[6]
                break
            index+=1
    return io_submiting,tp_IO_wait_time,tp_R_time,tp_W_time,tp_cp_time

def foward_trace(recv_queue:Queue, send_queue:Queue, pid:int,
                 io_submit_queue:Queue, io_finish_queue:Queue):
    # sleep_time = random.randint(0,GPU_NUM*10) # 差异化启动多个进程，模拟多GPU服务器繁忙程度不同的情况
    # time.sleep(sleep_time)
    # print("Init process-"+str(pid))
    listening1=True
    while True: # 持续处理，直到父进程通知其结束
        listening2=True
        while(listening2): # 持续接听，直到获取新任务
            delayMicrosecond(1000)
            if not recv_queue.empty():
                recv = recv_queue.get() # List:[session_id,q_id,stored_tokens]
                if len(recv) == 1:
                    assert recv[0] == -1
                    io_submit_queue.put([-1]) 
                    listening1=False
                else:
                    send_queue.put(["s",pid,recv[0]])
                listening2=False

        if listening1 == False:
            # print("Break pid="+str(pid))
            break
        
        comp_time=0
        IO_wait_time=0
        R_time=0
        W_time=0
        CP_time=0
        all_time = 0

        submit_comp_time=0
        submit_R_time=0
        submit_W_time=0

        TTFT = 0
        TPOT = 0
        generate_tokens_num = 0

        session_id=recv[0]
        q_id=recv[1]
        stored_tokens=recv[2]
        stored_tokens_k=stored_tokens
        stored_tokens_v=stored_tokens
        filepath=json_path+"session "+str(session_id)+" trace.json"

        io_submiting=[] #记录正在处理中的IO，格式为 [ ["R"或"W", "k"或"v", Gen_token_id, layer_id], [,,], [,,] ...]
        
        all_stime = time.time()
        with open(filepath,'r',encoding='utf8')as fp:
            json_datas = json.load(fp)
            assert q_id==json_datas[q_id]["q"]
            qinfo=json_datas[q_id]["I/O info"]
            load_next_layerid = -1
            load_next_token_len = -1
            
            for json_data in qinfo:
                # print(json_data)
                # if generate_tokens_num < int(json_data["Gen_token_id"]):
                #     print("session_id",session_id,"Gen_token_id:",json_data["Gen_token_id"])
                generate_tokens_num = int(json_data["Gen_token_id"])
                nameHead="S"+str(session_id)+"L"+str(json_data["layer_id"])

                if json_data["opration"] == "load weight":
                # 当前weight默认都在GPU中，不做卸载
                    pass

                elif json_data["opration"] == "compute":
                    stime = time.time()
                    R_stime = 0
                    # 检查 KVcache预取有没有完成
                    if json_data["layer_name"][-8:] == "selfattn":
                        wait = True
                        while wait and len(io_submiting)!=0:
                            wait = False
                            io_submiting, IO_wait_time, R_time, W_time, CP_time = check_io(io_finish_queue, io_submiting, IO_wait_time, R_time, W_time, CP_time)
                            for item in io_submiting:
                                if item[2] == json_data["Gen_token_id"] and item[3] == json_data["layer_id"]: # KVcache预取还没完成
                                    wait = True
                                    delayMicrosecond(100)
                                    # print("io_submiting:",io_submiting)
                                    break
                        
                        R_stime = time.time()
                        # 下一层 KVcache的预取
                        load_next_layerid = json_data["layer_id"] + 2
                        # 某些情况下不需要预取
                        if json_data["Gen_token_id"]>1 or q_id > 1:
                            # assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                            Gen_token_id = json_data["Gen_token_id"]
                            if load_next_layerid > MAX_ATT_LAYER_ID: # 如果是最后一层attention，则去取第一层的KVcache
                                load_next_layerid = 2
                                load_next_token_len=min(load_next_token_len+1, MAX_TOKENS) 
                                Gen_token_id=Gen_token_id+1
                            if load_next_token_len == -1:
                                print(json_data,load_next_layerid,load_next_token_len)
                                assert 0
                            io_submit_queue.put(["R", "k", Gen_token_id, session_id, load_next_layerid, load_next_token_len, time.time()]) 
                            io_submiting.append(["R","k", Gen_token_id, load_next_layerid])
                            io_submit_queue.put(["R", "v", Gen_token_id, session_id, load_next_layerid, load_next_token_len, time.time()]) 
                            io_submiting.append(["R","v", Gen_token_id, load_next_layerid])
                    if R_stime != 0:
                        R_etime = time.time()
                        submit_R_time += int((R_etime - R_stime)*1000000)
                    else:
                        R_etime = 0

                    sleep_time = int(float(json_data["time_cost(s)"])*1000000/GPU_SPEED_UP*LLM_SIZE_SCALING)
                    time_start=time.time() 
                    delayMicrosecond(sleep_time)
                    time_interval=int((time.time() - time_start)*1000000)
                    comp_time+=time_interval
                    submit_comp_time += int((time.time() - stime - ( R_etime-R_stime ))*1000000)

                    if int(json_data["Gen_token_id"]) == 1 and json_data["layer_name"] == "OutputEmbed":
                        TTFT = int((time.time() - all_stime)*1000000)

                elif json_data["opration"] == "store kcache" or json_data["opration"] == "store vcache":
                    stime = time.time()
                    # assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                    token_ids = []
                    token_len = int(json_data["session_len"])
                    for i in range(token_len-stored_tokens):
                        token_ids.append((i+stored_tokens) % MAX_TOKENS + 1)
                    # ["W", "k"或者"v", Gen_token_id, session_id, layer_id, io提交时间, token_ids:list ]
                    create_flag = True if q_id==1 and json_data["Gen_token_id"]==1 else False
                    io_submit_queue.put(["W", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], 
                                         session_id, json_data["layer_id"],token_ids,time.time(),create_flag]) 
                    io_submiting.append(["W", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], json_data["layer_id"]])

                    if json_data["layer_id"] == MAX_ATT_LAYER_ID: # store增量，#todo：异步IO适配
                        if str(json_data["opration"][-6:]) == "k":
                            stored_tokens_k=token_len
                        if str(json_data["opration"][-6:]) == "v":
                            stored_tokens_v=token_len
                        if stored_tokens_k==token_len and stored_tokens_v==token_len:
                            stored_tokens=token_len
                    submit_W_time += int((time.time() - stime)*1000000)

                elif json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache" or json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                    stime = time.time()
                    if json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache":
                        # assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                        token_len = min (int(json_data["session_len"])-1, MAX_TOKENS)
                    if json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                        # assert str(Max_KVcache_size)[1:-1] == str(json_data["shape"][1:-1])
                        token_len = min (int(json_data["history_len"]), MAX_TOKENS)
                    load_next_token_len = token_len
                    
                    # 仅当第一次 decode的第一层 KVcache load时无法与computation重叠
                    if json_data["layer_id"] == 2 and json_data["Gen_token_id"] == 1:
                        # pt_names=[]
                        # for i in range(token_len):
                        #     token_id = i+1
                        #     pt_names.append(KVPATH+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt")
                        # ["R", "k"或者"v", Gen_token_id, session_id, layer_id, token_len, io提交时间]
                        io_submit_queue.put(["R", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], 
                                            session_id, json_data["layer_id"], token_len, time.time()]) 
                        io_submiting.append(["R", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], json_data["layer_id"]])
                    submit_R_time += int((time.time() - stime)*1000000)

                else:
                    print("WARNING"+str(session_id)+"-"+str(q_id)+":\n"+str(json_data))
                    assert 0
        
        stime = time.time()
        # 将文件内容flush到磁盘并关闭mmap
        io_submit_queue.put(["F",-1,-1,-1,-1,-1,time.time()])
        io_submiting.append("F")
        submit_W_time += int((time.time() - stime)*1000000)
        while len(io_submiting) != 0:
            io_submiting, IO_wait_time, R_time, W_time, CP_time = check_io(io_finish_queue, io_submiting, IO_wait_time, R_time, W_time, CP_time)
            delayMicrosecond(100)

        all_time += int((time.time() - all_stime)*1000000)
        TPOT = (all_time-TTFT)/(generate_tokens_num-1)
  
        print("PrInfo: pid=",pid," s_id=",session_id," Q_id=",q_id,end="  ")
        print("all:",int(all_time/1000)/1000,"s",end="  ")
        print(" cpu:",int(comp_time/1000)/1000,"s",end="  ")
        print(" (IO_wait:",int(((IO_wait_time)/1000))/1000,"s",
                "  R:",int((R_time)/1000)/1000,"s",
                "  W:",int((W_time)/1000)/1000,"s",
                "  CP:",int((CP_time)/1000)/1000,"s)")
        print("      + submit_comp_time=",int(submit_comp_time/1000)/1000,"s",
                     " submit_R_time=",int(submit_R_time/1000)/1000,"s",
                     " submit_W_time=",int(submit_W_time/1000)/1000,"s")
        print("      + TTFT(mean):",int(TTFT/1000)/1000,"s",
                    "  TPOP(mean):",int(TPOT/1000)/1000,"s")

        send_queue.put(["e",session_id,pid,stored_tokens,comp_time,IO_wait_time,R_time,W_time,CP_time,all_time,TTFT,TPOT,generate_tokens_num])

def run__process():

    del_ptfile(KVPATH) # 删除 KVcache存储路径下的所有 KVcache

    # 多进程初始化
    manager =   Manager()
    send_queue  =   manager.Queue()
    recv_queue  =   manager.Queue()
    io_submit_queue = [manager.Queue()  for _ in range(GPU_NUM)]
    io_finish_queue = [manager.Queue()  for _ in range(GPU_NUM)]
    my_process      = []
    my_process_io   = []
    for i in range(GPU_NUM):
        my_process.append(Process(target=foward_trace, args=(send_queue,recv_queue,i,io_submit_queue[i],io_finish_queue[i])))
        my_process_io.append(Process(target=aio_trace, args=(io_submit_queue[i],io_finish_queue[i])))
    
    [p_io.start() for p_io in my_process_io]
    [p.start() for p in my_process]

    # 时间统计初始化
    all_comp_time=0
    all_IO_wait_time=0
    all_R_time=0
    all_W_time=0
    all_CP_time=0
    all_finish_qtime=0

    all_TTFT=0
    all_TPOT=0

    # 任务分发数据结构初始化
    session_nums=0 #共有几个session
    session_left_qnums=[] # 每个session还剩几个q没处理
    session_active=[] # 还没处理完的session的标识
    session_active_num=0
    session_processing=[] # 正在处理的session的标识
    session_stored_tokens=[] # 已存储完KVcache的token的id集合
    # session_tokens_id=[] # 一个二维list，存储着固定预算下，每个session当前保留的token id
    
    q_all_num = 0
    q_finish_num = 0 # 已处理完的Q的数量
    q_send_num = 0 # 已经发送的Q的数量

    with open(json_metapath,'r',encoding='utf8')as fp:
        json_datas = json.load(fp)
        session_nums=len(json_datas)
        session_active_num=len(json_datas)
        session_index=0
        for _ , qnum in json_datas.items():
            session_left_qnums.append(int(qnum))
            q_all_num += int(qnum)
            session_active.append(True)
            session_processing.append(False)
            session_stored_tokens.append(0)
            session_index+=1
        # print(session_left_qnums)
        print("session_nums=",session_nums)

    # 任务1开始分发
    stime=time.time()
    while session_active_num != 0:

        SEND_FLAG=False
        # 随机选取session部分：
        session_active_index = random.randint(0,session_nums-1) #从随机位置开始遍历
        search_list = [session_active_index+i for i in range(session_nums-session_active_index)]
        for i in range(session_active_index):
            search_list.append(i)
        session_id=-1
        for i in search_list:
            if session_active[i] == True: # 处于active的session才能被选中
                session_id = i
        if session_id == -1: # 没有session处于active，这与session_active_num != 0冲突
            assert 0

        if session_processing[session_id] == False: #若=True，此session正在被处理，请重新分发任务
            SEND_FLAG = True

        if MAX_Q != -1 and q_send_num == MAX_Q:
            SEND_FLAG = False
            if q_finish_num == MAX_Q:
                break
        
        if SEND_FLAG:
            filepath=json_path+"session "+str(session_id)+" trace.json"
            with open(filepath,'r',encoding='utf8')as fp:
                json_datas = json.load(fp)
                num_q=int(json_datas[0]["num_q"])
                qid = num_q-session_left_qnums[session_id] + 1
                if qid!=json_datas[qid]["q"]:
                    assert(0)

        while True: # 仅当send_queue中的任务被取完时，才会break出循环（见循环体的最后），派发新任务至send_queue
            while not recv_queue.empty() : # 子进程反馈信息优先处理，且必须处理完
                recv_list = recv_queue.get()

                if recv_list[0]=="s": # 子进程表示已接受任务
                    print("take  : s_id=",recv_list[2],"  pid=",recv_list[1])

                elif recv_list[0]=="e": # 子进程表示已完成任务
                    q_finish_num+=1        
                    tmp_session_id=recv_list[1]
                    session_stored_tokens[tmp_session_id]=recv_list[3]
                    all_comp_time+=recv_list[4]
                    all_IO_wait_time+=recv_list[5]
                    all_R_time+=recv_list[6]
                    all_W_time+=recv_list[7]
                    all_CP_time+=recv_list[8]
                    all_finish_qtime+=recv_list[9]
                    all_TTFT+=recv_list[10]
                    all_TPOT+=recv_list[11]

                    session_processing[tmp_session_id]=False
                    session_left_qnums[tmp_session_id]-=1
                    print("finish: s_id=",recv_list[1],"  pid=",tmp_session_id,"  q_finish_num=",q_finish_num,"(",q_all_num,")",end=" ")
                    if(session_left_qnums[tmp_session_id]==0):
                        session_active[tmp_session_id]=False
                        session_active_num-=1
                        # print("  session_left_qnums="+str(session_left_qnums),end=" ")
                        print(" => Deactivate:"+str(tmp_session_id),end=" ")
                    else:
                        # print("  session_left_qnums="+str(session_left_qnums),end=" ")
                        pass
                    print("")
                else:
                    assert 0

            if send_queue.empty() :
                break
            else:
                delayMicrosecond(500)

        if SEND_FLAG: # 派发新任务
            send_queue.put([session_id,qid,session_stored_tokens[session_id]])
            session_processing[session_id]=True
            print("public: s_id=", session_id, end=" ")
            # print(" session_left_qnums=", session_left_qnums)
            active_session_num = 0
            for active in session_active:
                if active:
                    active_session_num += 1 
            print("  active_session_num=",active_session_num)
            q_send_num += 1

    print("\nCPU:",int(all_comp_time/1000)/1000,"s",end="  ")
    print(" ( IO_wait:",int((all_IO_wait_time)/1000)/1000,"s",end=" ")
    print(" R:",int((all_R_time)/1000)/1000,"s",end=" ")
    print(" W:",int((all_W_time)/1000)/1000,"s",end=" ")
    print(" CP:",int((all_CP_time)/1000)/1000,"s)",end=" ")
    print(" all_qtime:",int((all_finish_qtime)/1000)/1000,"s")
    print(" TTFT(mean):",int((all_TTFT/q_send_num)/1000)/1000,"s",end=" ")
    print(" TPOP(mean):",int((all_TPOT/q_send_num)/1000)/1000,"s")
    print("ALL run_time",int((time.time() - stime)*1000)/1000,"s")
    

    for _ in range(GPU_NUM): # 结束子进程
        send_queue.put([-1])

    [p.join() for p in my_process_io]
    [p.join() for p in my_process]
  
if __name__ =='__main__':
    run__process()  # 正确做法：主线程只能写在 if内部


# # todo: mmap 的flush机制还不健全
# # todo：设定了静态的模型参数，不能自动读取模型元数据
# # 小文件太多，文件系统瓶颈导致IO写性能受限
# # pagecache缓存了大量KVcache，使读IO锐减，问题不突出
# # 暂不支持批处理