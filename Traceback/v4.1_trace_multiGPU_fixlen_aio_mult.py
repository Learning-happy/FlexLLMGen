import json
import time
import numpy as np
import torch
import random
from multiprocessing import Process,Pool,Pipe,Queue,Manager
from multiprocessing import Lock as mlock
from concurrent.futures import ThreadPoolExecutor

GPU_NUM = 2
PCIE_BW = 16 # GB/S
IO_THREAD_PER_LLM = 4

Max_TOKENS = 10 #最大KVcahe预算，当前实现了滑动窗口法
Max_KVcache_size=[543,32,64] # KVcache_data具体大小要根据模型参数手动指定来通过检查,否则assert
oneCache=torch.ones([1,32,64],dtype=torch.float16,device="cpu")
MaxAttLayerId=48

kvpath="./kvcache/"
# kvpath="/mnt/md0/" #需要先挂载RAID至该目录
json_metapath="../Tracefile/q_num_for_every_session.json"
json_path="../Tracefile/"

def delayMicrosecond(t):    # 微秒级延时函数
    start,end=0,0           # 声明变量
    start=time.time()       # 记录开始时间
    t=(t-0)/1000000     # 将输入t的单位转换为秒，-x是时间补偿
    while end-start<t:  # 循环至时间差值大于或等于设定值时
        end=time.time()     # 记录结束时间

def aio_thread_func(iotype:str, filepath:str):
    saveCache = oneCache
    data=[]
    if iotype == "R":
        data = torch.load(filepath, map_location=lambda storage, loc: storage,weights_only=True)
    if iotype == "W":
        torch.save(saveCache, filepath)
    return data

def aio_trace(io_submit_queue:Queue,io_submit_lock,
              io_finish_queue:Queue,io_finish_lock):
    
    saveCache = oneCache
    listening1=True
    with ThreadPoolExecutor(max_workers=IO_THREAD_PER_LLM) as executor:
        while True: # 持续处理，直到父进程通知其结束
            listening2=True
            while(listening2): # 持续接听，直到获取新任务
                io_submit_lock.acquire()
                if not io_submit_queue.empty():
                    recv = io_submit_queue.get()
                    if len(recv) == 1:
                        assert recv[0] == -1
                        listening1=False
                    listening2=False
                io_submit_lock.release()
                delayMicrosecond(50)

            if listening1 == False:
                break

            # 执行 IO
            # IO submit 格式：["R"或者"W", "k"或者"v", Gen_token_id, layer_id, [文件名，文件名，....]]
            all_copy_latency = 0
            copy_latency = (saveCache.numel() * saveCache.element_size()) / (PCIE_BW * 1000000000) * 1000000 # 以微妙为单位
            time_start=time.time()
            future_to_filepath = {executor.submit(aio_thread_func, recv[0],fp): fp for fp in recv[4]}
            for future in future_to_filepath:
                try:
                    future.result()
                    delayMicrosecond(copy_latency)
                    all_copy_latency += copy_latency
                except Exception as exc:
                    print(f"Failed to save to {future_to_filepath[future]}: {exc}")
            time_interval=int((time.time() - time_start)*1000000)

            # 返回结果
            io_finish_lock.acquire()
            # IO finish 格式：["R"或者"W", "k"或者"v", Gen_token_id, layer_id, IO时间 ]
            io_finish_queue.put([recv[0],recv[1],recv[2],recv[3],time_interval,all_copy_latency])
            io_finish_lock.release()

def check_io(io_finish_lock, io_finish_queue:Queue,io_submiting:list,R_time:int,W_time:int,CP_time:int):
    io_finish_lock.acquire()
    tp_R_time = R_time
    tp_W_time = W_time
    tp_cp_time = CP_time
    while not io_finish_queue.empty():
        # IO finish 格式：["R"或者"W", Gen_token_id, layer_id, IO时间 ]
        recv = io_finish_queue.get()
        index = 0
        for item in io_submiting:
            if item[0] == recv[0] and item[1] == recv[1] and item[2] == recv[2] and item[3] == recv[3]:
                io_submiting.pop(index) # 该IO结束，可以从异步等待队列移除
                if recv[0] == "R":
                    tp_R_time += recv[4]
                if recv[0] == "W":
                    tp_W_time += recv[4]
                tp_cp_time += recv[5]
                break
            index+=1
    io_finish_lock.release()
    return io_submiting,tp_R_time,tp_W_time,tp_cp_time

def foward_trace(recv_queue:Queue,  recv_lock,  send_queue:Queue,  send_lock, pid:int,
                 io_submit_queue:Queue, io_submit_lock, io_finish_queue:Queue, io_finish_lock):
    
    # print("Init process-"+str(pid))
    listening1=True
    while True: # 持续处理，直到父进程通知其结束
        listening2=True
        while(listening2): # 持续接听，直到获取新任务
            recv_lock.acquire()
            if not recv_queue.empty():
                recv = recv_queue.get() # List:[session_id,q_id,stored_tokens]
                if len(recv) == 1:
                    assert recv[0] == -1
                    io_submit_lock.acquire()
                    io_submit_queue.put([-1]) 
                    io_submit_lock.release()
                    listening1=False
                else:
                    send_queue.put(["s",pid,recv[0]])
                listening2=False
            recv_lock.release()

        if listening1 == False:
            # print("Break pid="+str(pid))
            break
        
        stime = time.time()
        comp_time=0
        R_time=0
        W_time=0
        CP_time=0

        session_id=recv[0]
        q_id=recv[1]
        stored_tokens=recv[2]
        stored_tokens_k=stored_tokens
        stored_tokens_v=stored_tokens
        filepath=json_path+"session "+str(session_id)+" trace.json"

        io_submiting=[] #记录正在处理中的IO，格式为 [ ["R"或"W", "k"或"v", Gen_token_id, layer_id], [,,], [,,] ...]
        
        with open(filepath,'r',encoding='utf8')as fp:
            json_datas = json.load(fp)
            assert q_id==json_datas[q_id]["q"]
            qinfo=json_datas[q_id]["I/O info"]
            
            for json_data in qinfo:
                # 处理一遍 IO结果
                io_submiting, R_time, W_time, CP_time = check_io(io_finish_lock, io_finish_queue, io_submiting, R_time, W_time, CP_time)
                nameHead="S"+str(session_id)+"L"+str(json_data["layer_id"])

                if json_data["opration"] == "load weight":
                # 当前weight默认都在GPU中，不做卸载
                    pass

                elif json_data["opration"] == "compute":
                    # 检查 KVcache预取有没有完成
                    if json_data["layer_name"][-8:] == "selfattn":
                        wait = True
                        while wait:
                            wait = False
                            io_submiting, R_time, W_time, CP_time = check_io(io_finish_lock, io_finish_queue, io_submiting, R_time, W_time, CP_time)
                            for item in io_submiting:
                                if item[2] == json_data["Gen_token_id"] and item[3] == json_data["layer_id"]: # KVcache预取还没完成
                                    wait = True

                    sleep_time = int(float(json_data["time_cost(s)"])*1000000)
                    time_start=time.time() 
                    delayMicrosecond(sleep_time)
                    time_interval=int((time.time() - time_start)*1000000)
                    comp_time+=time_interval

                elif json_data["opration"] == "store kcache" or json_data["opration"] == "store vcache":
                    assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                    pt_names=[]
                    token_len = int(json_data["session_len"])
                    for i in range(token_len-stored_tokens):
                        token_id = (i+stored_tokens) % Max_TOKENS + 1
                        pt_names.append(kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt")
                    io_submit_lock.acquire()
                    # ["R"或者"W", "k"或者"v", Gen_token_id, layer_id, [文件名，文件名，....]]
                    io_submit_queue.put(["W", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], 
                                         json_data["layer_id"], pt_names]) 
                    io_submit_lock.release()
                    io_submiting.append(["W", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], json_data["layer_id"]])

                    if json_data["layer_id"] == MaxAttLayerId: # store增量，#todo：异步IO适配
                        if str(json_data["opration"][-6:]) == "k":
                            stored_tokens_k=token_len
                        if str(json_data["opration"][-6:]) == "v":
                            stored_tokens_v=token_len
                        if stored_tokens_k==token_len and stored_tokens_v==token_len:
                            stored_tokens=token_len

                elif json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache" or json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                    if json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache":
                        assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                        token_len = min (int(json_data["session_len"])-1, Max_TOKENS)
                    if json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                        assert str(Max_KVcache_size)[1:-1] == str(json_data["shape"][1:-1])
                        token_len = min (int(json_data["history_len"]), Max_TOKENS)

                    pt_names=[]
                    for i in range(token_len):
                        token_id = i+1
                        pt_names.append(kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt")
                    io_submit_lock.acquire()
                    # ["R"或者"W", "k"或者"v", Gen_token_id, layer_id, [文件名，文件名，....]]
                    io_submit_queue.put(["R", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], 
                                         json_data["layer_id"], pt_names]) 
                    io_submit_lock.release()
                    io_submiting.append(["R", str(json_data["opration"][-6:-5]), json_data["Gen_token_id"], json_data["layer_id"]])

                else:
                    print("WARNING"+str(session_id)+"-"+str(q_id)+":\n"+str(json_data))
                    assert 0

        while len(io_submiting) != 0:
            io_submiting, R_time, W_time, CP_time = check_io(io_finish_lock, io_finish_queue, io_submiting, R_time, W_time, CP_time)

        print("  GPU=",pid," S_id=",session_id," Q_id=",q_id,end="  ")
        print("all:",int((time.time() - stime)*1000),"ms",end="  ")
        print("cpu:",int(comp_time/1000),"ms",end="  ")
        print("IO:",int(((R_time+W_time+CP_time)/1000)),"ms",end=" ")
        print("(R:",int((R_time)/1000),"ms",end="  ")
        print("W:",int((W_time)/1000),"ms",end="  ")
        print("CP:",int((CP_time)/1000),"ms)")

        send_lock.acquire()
        send_queue.put(["e",session_id,pid,stored_tokens,comp_time,R_time,W_time,CP_time])
        send_lock.release()

def run__process():

    # 多进程初始化
    manager =   Manager()
    send_queue  =   manager.Queue()
    send_lock   =   manager.Lock()
    recv_queue  =   manager.Queue()
    recv_lock   =   manager.Lock()
    io_submit_queue = [manager.Queue()  for _ in range(GPU_NUM)]
    io_submit_lock  = [manager.Lock()   for _ in range(GPU_NUM)]
    io_finish_queue = [manager.Queue()  for _ in range(GPU_NUM)]
    io_finish_lock  = [manager.Lock()   for _ in range(GPU_NUM)]
    my_process      = []
    my_process_io   = []
    for i in range(GPU_NUM):
        my_process.append(Process(target=foward_trace, args=(send_queue,send_lock,recv_queue,recv_lock,i,
                                                             io_submit_queue[i],io_submit_lock[i],io_finish_queue[i],io_finish_lock[i])))
        my_process_io.append(Process(target=aio_trace, args=(io_submit_queue[i],io_submit_lock[i],io_finish_queue[i],io_finish_lock[i])))
    
    [p_io.start() for p_io in my_process_io]
    [p.start() for p in my_process]

    # 时间统计初始化
    all_comp_time=0
    all_R_time=0
    all_W_time=0
    all_CP_time=0

    # 任务分发数据结构初始化
    session_nums=0 #共有几个session
    session_left_qnums=[] # 每个session还剩几个q没处理
    session_active=[] # 还没处理完的session的标识
    session_active_num=0
    session_processing=[] # 正在处理的session的标识
    session_stored_tokens=[] # 已存储完KVcache的token的id集合
    # session_tokens_id=[] # 一个二维list，存储着固定预算下，每个session当前保留的token id

    with open(json_metapath,'r',encoding='utf8')as fp:
        json_datas = json.load(fp)
        session_nums=len(json_datas)
        session_active_num=len(json_datas)
        session_index=0
        for _ , qnum in json_datas.items():
            session_left_qnums.append(int(qnum))
            session_active.append(True)
            session_processing.append(False)
            session_stored_tokens.append(0)
            session_index+=1
    print(session_left_qnums)
    print(session_active)

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
                recv_lock.acquire()
                recv_list = recv_queue.get()
                recv_lock.release()

                if recv_list[0]=="s": # 子进程表示已接受任务
                    print("take  : s_id="+str(recv_list[2]))

                elif recv_list[0]=="e": # 子进程表示已完成任务        
                    tmp_session_id=recv_list[1]
                    session_stored_tokens[tmp_session_id]=recv_list[3]
                    all_comp_time+=recv_list[4]
                    all_R_time+=recv_list[5]
                    all_W_time+=recv_list[6]
                    all_CP_time+=recv_list[7]

                    session_processing[tmp_session_id]=False
                    session_left_qnums[tmp_session_id]-=1
                    if(session_left_qnums[tmp_session_id]==0):
                        session_active[tmp_session_id]=False
                        session_active_num-=1
                        print("finish: s_id="+str(recv_list[1])+"  session_left_qnums="+str(session_left_qnums))
                        print("session_active changed:" + str(session_active) + "  deactivate:"+str(tmp_session_id))
                    else:
                        print("finish: s_id="+str(recv_list[1])+"  session_left_qnums="+str(session_left_qnums))
                    
                else:
                    assert 0

            if send_queue.empty() :
                break

        if SEND_FLAG: # 派发新任务
            send_lock.acquire()
            send_queue.put([session_id,qid,session_stored_tokens[session_id]])
            send_lock.release()
            session_processing[session_id]=True
            print("public: s_id="+str(session_id)+"  session_left_qnums="+str(session_left_qnums))

    print("\nCPU:",int(all_comp_time/1000),"ms",end="  ")
    print("IO:",int((all_R_time+all_W_time+all_CP_time)/1000),"ms",end=" ")
    print("(R:",int((all_R_time)/1000),"ms",end=" ")
    print("W:",int((all_W_time)/1000),"ms",end=" ")
    print("CP:",int((all_CP_time)/1000),"ms)")
    print("ALL run_time",int((time.time() - stime)*1000),"ms")
    
    send_lock.acquire()
    for _ in range(GPU_NUM): # 结束子进程
        send_queue.put([-1])
    send_lock.release()
    [p.join() for p in my_process_io]
    [p.join() for p in my_process]
  
if __name__ =='__main__':
    run__process()  # 正确做法：主线程只能写在 if内部


# # todo: 没有完全模拟出flexgen的IO方式（memmap）
# # todo：设定了静态的模型参数，不能自动读取模型元数据