import torch
import torchaudio
import os
import argparse
import resnet
import utils

# path = "./model_saved/model_save"

def parse_args():
    """
    return args
    """
    description = "use model predict result"                         
    parser = argparse.ArgumentParser(description=description)        
    parser.add_argument('-mp', '--modelpath',help = "path to saved model", default="./model_saved/model_save")  
    parser.add_argument('-op', '--outputpath', help="path of output", default="./pred/pred.out")
    parser.add_argument('-dp', '--datapath', help="path of data", default="../data/LibriSpeech-SI/test_wav")
    args = parser.parse_args()                                              
    return args




if __name__ == '__main__':
    args = parse_args()
    model=utils.load_model(args.modelpath)
    data=utils.get_data(args.datapath)
    print(data)
    print("output path:{0}".format(args.outputpath))
