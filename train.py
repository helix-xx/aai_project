import torch
from torch.utils.data import DataLoader
from torch import nn
import os
import data_process
import my_utils
import argparse
import resnet1D
import resnet2D
import time

def parse_args():
    """
    return args
    """
    description = "use model predict result"                         
    parser = argparse.ArgumentParser(description=description)        
    # parser.add_argument('-mp', '--modelpath',help = "path to saved model", default="./model_saved/model_save")  
    parser.add_argument('-sp', '--savepath', help="path of output", default="./model_saved/model_save_MeLspec")
    parser.add_argument('-dp', '--datapath', help="path of data", default="../data/LibriSpeech-SI/train_wav_aug")
    parser.add_argument('-np', '--noisepath', help="path of noisedata", default="../data/LibriSpeech-SI/noise_wav")
    parser.add_argument('-op', '--option', help="recognize option,FFT,MelSpec", default="MelSpec")
    args = parser.parse_args()                                              
    return args

def prepare_data(datapath,option,noisepath,device):
    if os.path.exists('X.pkl'):
        a=my_utils.load_data()
        X,Y=zip(*a)
    else:
        X,Y = my_utils.generate_origin(datapath)
        my_utils.save_data(X,Y)

    data_set = data_process.dataset(X,Y,option,noisepath,device)
    data_sizes=len(data_set)
    test_len=int(data_sizes*data_process.VALID_SPLIT)
    train_len=data_sizes-test_len
    train_dataset,test_dataset=torch.utils.data.random_split(data_set,[train_len,test_len],generator=torch.Generator().manual_seed(data_process.MANUAL_SEED))
    train_dataloader = DataLoader(train_dataset, batch_size=data_process.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=data_process.BATCH_SIZE, shuffle=True)
    return train_dataloader,test_dataloader


if __name__ == '__main__':
    args = parse_args()
    datapath = args.datapath
    option = args.option
    noisepath=my_utils.get_test_data(args.noisepath)

    if option == "FFT":
        model = resnet1D.ResNet()
        print("Use FFT to extract feature")
    else:
        option = "MelSpec"
        model = resnet2D.ResNet()
        print("Use MelSpec to extract feature")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    train_dataloader,test_dataloader = prepare_data(datapath,option,noisepath,device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = data_process.EPOCHS
    cnt=0
    for t in range(epochs//2):
        T1 =time.perf_counter()
        cnt+=1
        data_process.train(train_dataloader, model, loss_fn, optimizer, device)
        data_process.test(test_dataloader, model, loss_fn, device)
        print("epoch-{a},time used-{b}".format(a=cnt,b=time.perf_counter()-T1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for t in range(epochs//2):
        T1 =time.perf_counter()
        cnt+=1
        data_process.train(train_dataloader, model, loss_fn, optimizer, device)
        data_process.test(test_dataloader, model, loss_fn, device)
        print("epoch-{a},time used-{b}".format(a=cnt,b=time.perf_counter()-T1))
    torch.save(model.state_dict(),args.savepath)