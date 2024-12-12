import json
import time
import numpy as np
import torch
import random
from multiprocessing import Process,Pool,Pipe,Queue,Manager
from multiprocessing import Lock as mlock

GPU_NUM = 4

Max_TOKENS = 10 #最大KVcahe预算，当前实现了滑动窗口法
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

def foward_trace(recv_queue:Queue,  recv_lock,
                 send_queue:Queue,  send_lock,
                 pid:int):
    
    # print("Init process-"+str(pid))
    listening1=True
    while True: # 持续处理，直到父进程通知其结束
        listening2=True
        while(listening2): # 持续接听，直到获取新任务
            recv_lock.acquire()
            if recv_queue.empty() != True:
                recv = recv_queue.get() # List:[session_id,q_id,stored_tokens]
                if len(recv) == 1:
                    assert recv[0] == -1
                    listening1=False
                else:
                    send_queue.put(["s",pid,recv[0]])
                listening2=False
            recv_lock.release()

        if listening1 == False:
            # print("Break pid="+str(pid))
            break

        finish_time=0
        R_time=0
        W_time=0
        comp_time=0

        session_id=recv[0]
        q_id=recv[1]
        stored_tokens=recv[2]
        stored_tokens_k=stored_tokens
        stored_tokens_v=stored_tokens
        filepath=json_path+"session "+str(session_id)+" trace.json"
        
        with open(filepath,'r',encoding='utf8')as fp:
            json_datas = json.load(fp)
            assert q_id==json_datas[q_id]["q"]
            qinfo=json_datas[q_id]["I/O info"]
            
            all_time_start=time.time()
            for json_data in qinfo:
                nameHead="S"+str(session_id)+"L"+str(json_data["layer_id"])

                if json_data["opration"] == "load weight":
                # 当前weight默认都在GPU中，不做卸载
                    pass

                elif json_data["opration"] == "compute":
                    sleep_time = int(float(json_data["time_cost(s)"])*1000000)
                    time_start=time.time() 
                    delayMicrosecond(sleep_time)
                    time_interval=int((time.time() - time_start)*1000000)
                    # timePrint("\n  compute_time="+str(sleep_time)+" -- "+str(time_interval))
                    comp_time+=time_interval

                elif json_data["opration"] == "store kcache" or json_data["opration"] == "store vcache":
                    assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                    # token_len=int(json_data["session_len"])
                    token_len = int(json_data["session_len"])
                    for i in range(token_len-stored_tokens):
                        token_id = (i+stored_tokens) % Max_TOKENS + 1
                        pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                        time_start=time.time()
                        torch.save(oneCache, pt_name)
                        time_interval=int((time.time() - time_start)*1000000)
                        # timePrint("  "+str(json_data["opration"][-6:-5])+"_saveTime="+str(time_interval))
                        W_time+=time_interval
                    if json_data["layer_id"] == MaxAttLayerId: # store增量，#todo：异步IO适配
                        if str(json_data["opration"][-6:]) == "k":
                            stored_tokens_k=token_len
                        if str(json_data["opration"][-6:]) == "v":
                            stored_tokens_v=token_len
                        if stored_tokens_k==token_len and stored_tokens_v==token_len:
                            stored_tokens=token_len

                elif json_data["opration"] == "load kcache" or json_data["opration"] == "load vcache":
                    assert str(Max_KVcache_size) == str(json_data["shape"][11:-1])
                    token_len = min (int(json_data["session_len"])-1, Max_TOKENS)
                    # print("R-io"+str(token_len))
                    for i in range(token_len):
                        token_id = i+1
                        pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                        time_start=time.time() 
                        data = torch.load(pt_name, map_location=lambda storage, loc: storage,weights_only=True)
                        time_interval=int((time.time() - time_start)*1000000)
                        # timePrint("  "+str(json_data["opration"][-6:-5])+"_loadTime="+str(time_interval))
                        R_time+=time_interval

                elif json_data["opration"] == "load history kcache" or json_data["opration"] == "load history vcache":
                    assert str(Max_KVcache_size)[1:-1] == str(json_data["shape"][1:-1])
                    token_len = min (int(json_data["history_len"]), Max_TOKENS)
                    # print("R-io"+str(token_len))
                    for i in range(token_len):
                        token_id = i+1
                        pt_name = kvpath+nameHead+"T"+str(token_id)+str(json_data["opration"][-6:])+".pt"
                        time_start=time.time() 
                        data = torch.load(pt_name, map_location=lambda storage, loc: storage,weights_only=True)
                        time_interval=int((time.time() - time_start)*1000000)
                        # timePrint("  his_"+str(json_data["opration"][-6:-5])+"_load_time="+str(time_interval))
                        R_time+=time_interval

                else:
                    print("WARNING"+str(session_id)+"-"+str(q_id)+":\n"+str(json_data))
                    assert 0

        finish_time = int((time.time() - all_time_start)*1000000)

        print("  GPU="+str(pid)+" S_id="+str(session_id)+" Q_id="+str(q_id),end="  ")
        print("ALL_time:"+str(finish_time/1000)+" ms",end="  ")
        print("CPU_time:"+str((comp_time)/1000)+"ms",end="  ")
        print("IO_time:"+str((R_time+W_time)/1000)+"ms",end="  ")
        print("R_time:"+str((R_time)/1000)+"ms",end="  ")
        print("W_time:"+str((W_time)/1000)+"ms")

        send_lock.acquire()
        send_queue.put(["e",session_id,pid,stored_tokens,finish_time,R_time,W_time])
        send_lock.release()

def run__process():

    # 多进程初始化
    manager=Manager()
    send_queue=manager.Queue()
    send_lock=manager.Lock()
    recv_queue=manager.Queue()
    recv_lock=manager.Lock()
    my_process=[]
    for i in range(GPU_NUM):
        my_process.append(Process(target=foward_trace, args=(send_queue,send_lock,recv_queue,recv_lock, i)))
    [p.start() for p in my_process]

    # 时间统计初始化
    all_finish_time=0
    all_R_time=0
    all_W_time=0

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
            while recv_queue.empty() != True: # 子进程反馈信息优先处理，且必须处理完
                recv_lock.acquire()
                recv_list = recv_queue.get()
                recv_lock.release()

                if recv_list[0]=="s": # 子进程表示已接受任务
                    print("take  : s_id="+str(recv_list[2]))

                elif recv_list[0]=="e": # 子进程表示已完成任务        
                    tmp_session_id=recv_list[1]
                    session_stored_tokens[tmp_session_id]=recv_list[3]
                    all_finish_time+=recv_list[4]
                    all_R_time+=recv_list[5]
                    all_W_time+=recv_list[6]

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

    print("\n\nAll over!!!!!!!!\n")
    print("ALL_time:"+str(all_finish_time/1000)+" ms",end="  ")
    print("IO_time:"+str((all_R_time+all_W_time)/1000)+"ms",end="  ")
    print("R_time:"+str((all_R_time)/1000)+"ms",end="  ")
    print("W_time:"+str((all_W_time)/1000)+"ms")
    
    send_lock.acquire()
    for _ in range(GPU_NUM): # 结束子进程
        send_queue.put([-1])
    send_lock.release()
    [p.join() for p in my_process]

  
if __name__ =='__main__':
    stime=time.time()
    run__process()  # 正确做法：主线程只能写在 if内部
    print("run_time",int((time.time() - stime)*1000)," ms")


# # todo: 没有模拟出flexgen的IO方式（异步IO）
# # todo：没有考虑tensor从内存拷贝到GPU的时间