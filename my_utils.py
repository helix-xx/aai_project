import torch
import torchaudio
import os
import resnet1D
import sklearn
from sklearn import preprocessing
import pickle



def generate_origin(path="/home/lizz_lab/cse12232433/project/aai_project/data/LibriSpeech-SI/train_wav"):
    '''generate origin dataset,need convert by dataset() and dataloader()'''
    # DATASET_AUDIO_PATH = os.path.join(root_path,traindir)
    # file name is the speaker name
    class_names = os.listdir(path)
    X=[];Y=[];
    def listdir_addXY(path, labels, X, Y):
        for label in labels:
            for file in os.listdir(path+'/'+label):
                file_path = os.path.join(path+'/'+label, file)
                if os.path.isdir(file_path):
                    listdir_addXY(file_path, X, Y)
                else:
                    X.append(file_path)
                    Y.append(label)
    listdir_addXY(path,class_names,X,Y)
    Y = torch.as_tensor(preprocessing.LabelEncoder().fit_transform(Y))
    return X,Y

def save_data(X, Y, path="/home/lizz_lab/cse12232433/project/aai_project/test_model/"):
    '''use pickle save train data, X is the path of audio, Y is label'''
    with open(path+"X.pkl",mode="wb") as f:
        pickle.dump(X, f)
    with open(path+"Y.pkl",mode="wb") as f:
        pickle.dump(Y, f)

def load_data(path="/home/lizz_lab/cse12232433/project/aai_project/test_model/"):
    '''load saved train data'''
    if os.path.exists(path+'X.pkl'):
        X=pickle.load(open(path+'X.pkl','rb'))
    if os.path.exists(path+'Y.pkl'):
        Y=pickle.load(open(path+'Y.pkl','rb'))
    res=zip(X,Y)
    return res

def get_test_data(path):
    '''Get data_path to test, need convert before to model'''
    print("get_data")
    all_path=[]
    for file in os.listdir(path):
        file = os.path.join(path+'/', file)
        if os.path.isdir(file):
            get_test_data(file)
        else:
            all_path.append(file)
    return all_path

def load_model(path, device):
    '''load trained model'''
    print("load model")
    model = resnet1D.ResNet()
    if device=="CPU":
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(path))
    return model

def remove_file(path="../data/LibriSpeech-SI/noise_wav"):
    all_path=get_test_data(path)
    for path in all_path:
        try:
            audio,sample_rate=torchaudio.load(path)
        except:
            print("remove file{a}".format(a=path))
            os.remove(path)

def print_cuda_info():
    print("Number of devices:{a}*{b}. \n".format(a=torch.cuda.device_count(),b=torch.cuda.get_device_name()))