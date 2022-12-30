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
    parser.add_argument('-mp', '--modelpath',help = "path to saved model", default="./model_saved/model_save_MeLspec640")  
    parser.add_argument('-op', '--outputpath', help="path of output", default="./pred/pred_noise.txt")
    parser.add_argument('-dp', '--datapath', help="path of data", default="../data/LibriSpeech-SI/test-noisy_wav")
    parser.add_argument('-opt', '--option', help="recognize option,FFT,MelSpec", default="MelSpec")
    args = parser.parse_args()                                              
    return args


if __name__ == '__main__':
    args = parse_args()
    modelpath=args.modelpath
    datapath=args.datapath
    outputpath=args.outputpath
    option=args.option

    if option == "FFT":
        print("Use FFT to extract feature")
    else:
        option = "MelSpec"
        print("Use MelSpec to extract feature")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("work on {a}".format(a=device))
    model=my_utils.load_model(modelpath, device, option)
    model.eval()
    # if torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model)

    data_paths=my_utils.get_test_data(datapath)
    
    res = open(outputpath, mode = 'a',encoding='utf-8')
    for path in data_paths:
        # path = data_paths[1]
        print(path)
        audio = data_process.path_to_audio(path)
        audio = data_process.fixed_length(audio)
        audio = audio.to(device)
        if option == "FFT":
            data = data_process.audio_fft(audio)
            data = data.unsqueeze(dim=-2)
        elif option == "MelSpec":
             data = data_process.audio_melspec(audio,device)
             data = data.unsqueeze(dim=0)
        # data = data_process.audio_to_fft(audio)
        # data need in batch format 1,1,6400
        # print(data.size())
        # pred = model(data).argmax(1)
        pred = model(data)
        # a, b = torch.sort(pred, descending=True)
        # print(a,b)
        pred = pred.argmax(1)
        print(path.split('/')[-1]+' '+"spk"+str(pred.item()+1),file=res)
        # break
    res.close()