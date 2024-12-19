import json
import os
import subprocess

test_path = "./batch_test/"

def get_system_info():
    # 获取CPU核心数量
    cpu_cores = os.cpu_count()
    # 获取内存大小
    mem_info = subprocess.check_output(['cat', '/proc/meminfo']).decode('utf-8')
    mem_total = None
    for line in mem_info.split('\n'):
        if line.startswith('MemTotal:'):
            mem_total = int(int(line.split()[1]) / 1024 /1024*1000)/1000  # 转换为MB
            break
    
    # 获取磁盘类型和大小
    disk_info = subprocess.check_output(['lsblk', '-o', 'NAME,FSTYPE,SIZE']).decode('utf-8')
    disks = {}
    for line in disk_info.split('\n'):
        parts = line.split()
        if len(parts) == 3:
            disk_name, disk_type, disk_size = parts
            if disk_name[:4]!= "loop":
                disks[disk_name] = {'type': disk_type, 'size': disk_size}
    
    return {
        'CPU Cores': cpu_cores,
        'Memory Size': f'{mem_total} GB',
        'Disks': disks
    }

test_path = "./batch_test/"
logfile_num = 0
logfile_date = []
with open("meta_config.json", 'r', encoding='utf-8') as mf:
    json_datas = json.load(mf)
    with open("meta_merge.log", 'w', encoding='utf-8') as ml:
        system_info = get_system_info()
        ml.write(str("system_info:\n"))
        json.dump(system_info,ml,ensure_ascii=False, indent=4)
        ml.write("\n\n")
        log_list = os.listdir(test_path)
        for f in log_list:
            file_path = os.path.join(test_path, f)
            if os.path.isfile(file_path) and str(file_path)[-4:]==".tpl":
                logfile_num+=1
        for i in range(logfile_num):
            file_path = test_path+str(i)+".tpl"
            # print("file_path:",file_path)
            with open(file_path, 'r', encoding='utf-8') as fp:
                logfile_date.append(json.load(fp))
                
        for i in range(logfile_num):
            ml.write("ALL_run_time:"+str(logfile_date[i]["all_time"])+"s  ")
            ml.write("TTFT(mean):"+str(logfile_date[i]["TTFT"])+"s  ")
            ml.write("TPOP(mean):"+str(logfile_date[i]["TPOT"])+"s\n")
            ml.write("CPU:"+str(logfile_date[i]["comp_time"])+"s  ")
            ml.write("( IO_wait:"+str(logfile_date[i]["IO_wait_time"])+"s  ")
            ml.write("R:"+str(logfile_date[i]["R_time"])+"s  ")
            ml.write("W:"+str(logfile_date[i]["W_time"])+"s  ")
            ml.write("CP:"+str(logfile_date[i]["CP_time"])+"s)  ")
            ml.write("=> all_qtime:"+str(logfile_date[i]["all_q_time"])+"s\n")

            ml.write("ALL_io_submit    MB_R="+str(logfile_date[i]["submit_r_size"])+"  MB_w="+str(logfile_date[i]["submit_w_size"])+"\n")
            for j in range(len(logfile_date[i]["device"])):
                ml.write(str(logfile_date[i]["device"][j])+"  MB_read="+str(logfile_date[i]["MB_read"][j])+"\t MB_wrtn="+str(logfile_date[i]["MB_wrtn"][j])+"\n")
            ml.write(str(json_datas[i]))
            ml.write("\n\n")
            if (i+1) %2 == 0:
                ml.write("................................................."+"\n\n")

with open("meta_log.json", 'w', encoding='utf-8') as fp:
    json.dump(logfile_date,fp,ensure_ascii=False, indent=4)
            



