import os

# flac to wav all
need_convert=["noise","test","test-noisy","train"]
root_path="/home/lizz_lab/cse12232433/project/aai_project/data/LibriSpeech-SI/"

# for _ in need_convert:
#     print(root_path+_+'/')
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
        convertpath = root_path+convert+"_wav/"+originpath.split('.')[0]+".wav"
        originpath = root_path+convert +"/" + originpath
        if dir != "/".join(convertpath.split('/')[0:-1]):
            dir = "/".join(convertpath.split('/')[0:-1])
            if os.path.exists(dir) is False:
                os.makedirs(dir)
        if os.path.exists(convertpath) is False:
            os.system('ffmpeg -i %s %s > /dev/null' % (originpath,convertpath))
            y_cnt=y_cnt+1
        else:
            n_cnt=n_cnt+1
    print("正在转换%s,已完成%s,共计%s" % (item_name,y_cnt,sizes))
