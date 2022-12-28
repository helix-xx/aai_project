import torch
import torchaudio
import os
import argparse
import my_utils
import data_process
# path = "./model_saved/model_save"

def parse_args():
    """
    return args
    """
    description = "use model predict result"                         
    parser = argparse.ArgumentParser(description=description)        
    parser.add_argument('-mp', '--modelpath',help = "path to saved model", default="./model_saved/model_save")  
    parser.add_argument('-op', '--outputpath', help="path of output", default="./pred/pred.txt")
    parser.add_argument('-dp', '--datapath', help="path of data", default="../data/LibriSpeech-SI/test_wav")
    args = parser.parse_args()                                              
    return args


if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("work on {a}".format(a=device))
    model=my_utils.load_model(args.modelpath, device)
    # if torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model)

    data_paths=my_utils.get_test_data(args.datapath)
    
    res = open(args.outputpath, mode = 'a',encoding='utf-8')
    for path in data_paths:
        print(path)
        audio = data_process.path_to_audio(path)
        audio = data_process.fixed_length(audio)
        data = data_process.audio_to_fft(audio)
        # data need in batch format 1,1,6400
        data = data.unsqueeze(dim=-2)
        data = data.unsqueeze(dim=-2)
        data = data.to(device)
        pred = model(data).argmax(1)
        print(path.split('/')[-1]+' '+"spk"+str(pred.item()+1),file=res)
    res.close()
