import json
import os

def del_json_file(filepath,tail:str):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path) and str(file_path)[-1*len(tail):]==tail:
            os.remove(file_path)

# 目标GPU与原GPU的推理性能之比（原GPU：录制trace文件的GPU），默认为 1
# 4090:2 V100:3 A100:8 H800:50
GPU_SPEED_UP = [1,5] 
# [不推荐使用] 目标LLM与原LLM的大小之比（原LLM：录制trace文件的LLM），默认为 1
# 注意:模型层数无法改变，仅scale了每层的参数量
LLM_SIZE_SCALING = [1,5,10] 
GPU_NUM = [1,2,4] # 模拟的GPU（LLM实例）数量
MAX_Q = [-1] # Trace回放数量，以Q为单位：考虑到测试集太大，可以推理到一定的数量就停止回放；默认值-1，即全部测试完

# TRACE_FILE_PARAMETERS = [32,64,48,24,"/home/femu/FlexLLMGen/Tracefile/",1] # opt1.3B
# TRACE_FILE_PARAMETERS = [32,64,48,24,"/home/femu/MoreTracefile/101sessions_opt1-3/Tracefile/",1] # opt1.3B
# TRACE_FILE_PARAMETERS = [32,64,48,24,"/home/femu/MoreTracefile/6sessions_opt1.3_batch8/",8] # opt1.3B
# TRACE_FILE_PARAMETERS = [32,128,64,32,"/home/femu/MoreTracefile/6sessions_opt6.7_batch1/",1] # opt6.7B
# TRACE_FILE_PARAMETERS = [32,128,64,32,"/home/femu/MoreTracefile/6sessions_opt6.7_batch8/",8] # opt6.7B
TRACE_FILE_PARAMETERS =[[32,64,48,24,"/home/femu/MoreTracefile/6sessions_opt1.3_batch8/",8],[32,128,64,32,"/home/femu/MoreTracefile/6sessions_opt6.7_batch8/",8]]
MAX_TOKENS = [1024,2048] #最大KVcahe预算，当前实现了滑动窗口法

# KVCACHE_STORE_INFO = ["./kvcache/",["sda2"]]
# KVCACHE_STORE_INFO = ["/mnt/md0/kvcache/", ["md0","nvme0n1","nvme1n1","nvme2n1","nvme3n1"]] #需要先挂载 RAID 至/mnt/md0/
# KVCACHE_STORE_INFO = ["/mnt/ssd/kvcache/", ["nvme4n1"]] #需要先挂载 femu盘 至/mnt/ssd/
KVCACHE_STORE_INFO = [["/mnt/md0/kvcache/", ["md0","nvme0n1","nvme1n1","nvme2n1","nvme3n1"]], ["/mnt/ssd/kvcache/", ["nvme4n1"]]]

test_path = "./batch_test/"

del_json_file(test_path,".tpj")
del_json_file(test_path,".tpl")
del_json_file(test_path,".printlog")
del_json_file(test_path,".log")
del_json_file(test_path,".json")

test_id=0
with open("meta_config.json", 'w', encoding='utf-8') as mf:
    datas = []
    for gsu in GPU_SPEED_UP:
        for lss in LLM_SIZE_SCALING:
            for gn in GPU_NUM:
                for mq in MAX_Q:
                    for _ in [0]:
                        for _ in [0]:
                            for tfp in TRACE_FILE_PARAMETERS:
                                for mt in MAX_TOKENS:
                                    for ksi in KVCACHE_STORE_INFO:
                                        filename = test_path + str(test_id)+".tpj"
                                        # 写入JSON文件
                                        data = {
                                            "test_id" : test_id,
                                            "GPU_SPEED_UP" : gsu, 
                                            "LLM_SIZE_SCALING" : lss, 
                                            "GPU_NUM" : gn,
                                            "MAX_Q" : mq,
                                            "TRACE_FILE_PARAMETERS": tfp,
                                            "MAX_TOKENS" : mt,
                                            "KVCACHE_STORE_INFO": ksi,
                                            "log_path": test_path+str(test_id)+".tpl"
                                        }
                                        datas.append(data)
                                        with open(filename, 'w', encoding='utf-8') as f:
                                            json.dump(data, f, ensure_ascii=False, indent=4)
                                        print("echo -e \"test_id =",test_id,"  time=","\\c\"")
                                        print("date")
                                        print("python3 /home/femu/FlexLLMGen/Traceback/trace.py "+test_path+str(test_id)+".tpj"" > "+test_path+str(test_id)+".printlog")
                                        test_id += 1
    json.dump(datas, mf, ensure_ascii=False, indent=4)
