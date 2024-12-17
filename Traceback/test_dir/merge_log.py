import json
import os

test_path = "./batch_test/"

logfile_num = 0
logfile_date = []
with open("meta.json", 'r', encoding='utf-8') as mf:
    json_datas = json.load(mf)
    with open("merge.log", 'w', encoding='utf-8') as ml:
        log_list = os.listdir(test_path)
        for f in log_list:
            file_path = os.path.join(test_path, f)
            if os.path.isfile(file_path) and str(file_path)[-4:]==".log":
                logfile_num+=1
                logfile_date.append(["【lack】\n","【lack】\n","【lack】\n","【lack】\n"])
        for i in range(logfile_num):
            file_path = test_path+str(i)+".log"
            # print("file_path:",file_path)
            with open(file_path, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                # 获取最后三行（注意：如果文件行数少于3行，这里会返回所有行）
                if len(lines) >= 3:
                    logfile_date[i][0] = str(i)+".log"
                    logfile_date[i][1] = lines[-3]
                    logfile_date[i][2] = lines[-2]
                    logfile_date[i][3] = lines[-1]
                else:
                    print("WARNING:len(lines)=",len(lines)," file_path=",file_path)
                    # assert 0
                    logfile_date[i][0] = str(i)+".log"
                
        for i in range(logfile_num):
            # ml.write("【"+str(logfile_date[i][0])+"】  ")
            ml.write(str("    ")+str(logfile_date[i][3]))
            ml.write(str("    ")+str(logfile_date[i][1]))
            ml.write(str("   ")+str(logfile_date[i][2]))
            ml.write(str("    ")+str(json_datas[i])+"\n\n")
            



