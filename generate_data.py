import data_process
import my_utils
import torch
import os
# generate datafile
# 1.05 0.95 加减速 增加噪声，复制切片成10个文件
root_path="/home/lizz_lab/cse12232433/project/aai_project/data/LibriSpeech-SI/"
need_convert=["train_wav"]

def listdir(path, list_name,prefix_len):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name,prefix_len)
        else:
            list_name.append(file_path[prefix_len:])

# listdir(root_path+need_convert[3]+'/',all_path)
for convert in need_convert:
    all_path=[]
    listdir(root_path+convert+'/',all_path,len(root_path+convert+'/'))
    sizes=len(all_path)
    item_name=convert
    y_cnt=0
    n_cnt=0
    for originpath in all_path:
        convertpath = root_path+convert+"_aug/"+originpath.split('.')[0]+".wav"
        originpath = root_path+convert +"/" + originpath
        if dir != "/".join(convertpath.split('/')[0:-1]):
            dir = "/".join(convertpath.split('/')[0:-1])
            if os.path.exists(dir) is False:
                os.makedirs(dir)
        if os.path.exists(convertpath) is False:
            # os.system('ffmpeg -i %s %s > /dev/null' % (originpath,convertpath))
            data=data_process.path_to_audio(originpath)
            a,b=data_process.change_speed(data)
            data=torch.cat((a,data,b),0)
            data=data_process.fixed_length(data,41*16000)
            data_process.cut_save_audio(data,convertpath)
            y_cnt=y_cnt+1
        else:
            n_cnt=n_cnt+1
    print("正在转换%s,已完成%s,共计%s" % (item_name,y_cnt,sizes),flush=True)