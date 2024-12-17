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
GPU_SPEED_UP = [1,3,8] 
# [不推荐使用] 目标LLM与原LLM的大小之比（原LLM：录制trace文件的LLM），默认为 1
# 注意:模型层数无法改变，仅scale了每层的参数量
LLM_SIZE_SCALING = [1,5,20] 

GPU_NUM = [1,2,4] # 模拟的GPU（LLM实例）数量
MAX_Q = [-1] # Trace回放数量，以Q为单位：考虑到测试集太大，可以推理到一定的数量就停止回放；默认值-1，即全部测试完
BATCH_SIZE = [1,8,32]
batch_slowdown = [1] # batch_size的大小对计算开销的影响模拟

HEAD_NUM = [32] #注意力头数，需根据模型手动指定，当前为 opt1.3B
WEIGHT_DEM = [64] #权重维度，需根据模型手动指定，当前为 opt1.3B
MAX_ATT_LAYER_ID=[48] # attention的最后一层在全局的 Layer Id，需根据模型手动指定，当前为 opt1.3B
ATT_LAYER_NUM=[24] # attention的最后一层在全局的 Layer Id，需根据模型手动指定，当前为 opt1.3B
# json_path = "/home/femu/FlexLLMGen/Tracefile/","/home/femu/MoreTracefile/101sessions_opt1-3/Tracefile/"
json_path = ["/home/femu/FlexLLMGen/Tracefile/"]

MAX_TOKENS = [512] #最大KVcahe预算，当前实现了滑动窗口法

# KVPATH = "./kvcache/"、"/mnt/md0/kvcache/" 、"/mnt/ssd/kvcache/"
KVPATH = ["/mnt/md0/kvcache/", "/mnt/ssd/kvcache/"]

test_path = "./batch_test/"

del_json_file(test_path,".json")
del_json_file(test_path,".log")

test_id=0
with open("meta.json", 'w', encoding='utf-8') as mf:
    datas = []
    for gsu in GPU_SPEED_UP:
        for lss in LLM_SIZE_SCALING:
            for gn in GPU_NUM:
                for mq in MAX_Q:
                    for bs in BATCH_SIZE:
                        for bsd in batch_slowdown:
                            for i in range(len(HEAD_NUM)):
                                hn = HEAD_NUM[i]
                                wd = WEIGHT_DEM[i]
                                mali = MAX_ATT_LAYER_ID[i]
                                aln = ATT_LAYER_NUM[i]
                                jp = json_path[i]
                                for mt in MAX_TOKENS:
                                    for kvp in KVPATH:
                                        filename = test_path + str(test_id)+".json"
                                        # 写入JSON文件
                                        data = {
                                            "test_id" : test_id,
                                            "GPU_SPEED_UP" : gsu, 
                                            "LLM_SIZE_SCALING" : lss, 
                                            "GPU_NUM" : gn,
                                            "MAX_Q" : mq,
                                            "BATCH_SIZE" : bs,
                                            "batch_slowdown" : bsd,
                                            "HEAD_NUM" : hn,
                                            "WEIGHT_DEM" : wd,
                                            "MAX_ATT_LAYER_ID" : mali,
                                            "ATT_LAYER_NUM" : aln,
                                            "MAX_TOKENS" : mt,
                                            "KVPATH" : kvp,
                                            "json_path" : jp,
                                        }
                                        datas.append(data)
                                        with open(filename, 'w', encoding='utf-8') as f:
                                            json.dump(data, f, ensure_ascii=False, indent=4)
                                        print("echo -e \"test_id =",test_id,"  time=","\\c\"")
                                        print("date")
                                        print("python3 /home/femu/FlexLLMGen/Traceback/trace.py "+test_path+str(test_id)+".json"" > "+test_path+str(test_id)+".log")
                                        test_id += 1
    json.dump(datas, mf, ensure_ascii=False, indent=4)
