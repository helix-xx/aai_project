import torch
import os
import resnet
import sklearn
from sklearn import preprocessing


def generate_origin(root_path="/home/lizz_lab/cse12232433/project/aai_project/data/LibriSpeech-SI/",traindir="train_wav"):
    '''generate origin dataset,need convert by dataset() and dataloader()'''
    DATASET_AUDIO_PATH = os.path.join(root_path,traindir)
    # file name is the speaker name
    class_names = os.listdir(DATASET_AUDIO_PATH)
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
    listdir_addXY(DATASET_AUDIO_PATH,class_names,X,Y)
    Y = torch.as_tensor(preprocessing.LabelEncoder().fit_transform(Y))
    return X,Y

def get_data(path):
    '''Get data_path to test, need convert before to model'''

    print("get_data")
    all_path=[]
    for file in os.listdir(path):
        if os.path.isdir(file):
            get_data(file)
        else:
            all_path.append(file)
    return all_path

def load_model(path):
    '''load trained model'''

    print("load model")
    model = resnet.ResNet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    # model.eval()
    return model


def fixed_length(audio):
    if audio.size()[0] >= LENGTH:
        return audio[0:LENGTH]
    else:
        audio = torch.cat((audio,audio),0)
        return fixed_length(audio) 

def print_test():
    print("gogogo")
    return 1