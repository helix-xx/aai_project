import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import random
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# parameters
VALID_SPLIT = 0.2
MANUAL_SEED = 42
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000
BATCH_SIZE = 512
EPOCHS = 50
LENGTH = 4*16000

# torch.random.manual_seed(MANUAL_SEED)
def rms_db(data):
    mean_square = torch.mean(data**2)
    return 10 * torch.log10(mean_square)

def add_noise(noise_paths,wav,p=0.3,SNR=15):
    # if random.random()>p: return wav
    idx = random.randint(0,len(noise_paths)-1)
    noise = path_to_audio(noise_paths[idx])
    noise_gain_db = rms_db(wav) - rms_db(noise) - SNR
    noise *= 10. ** (noise_gain_db / 20.)
    noise_new = torch.zeros(wav.shape, dtype=torch.float32)
    if noise.shape[0] >= wav.shape[0]:
        start = random.randint(0, noise.shape[0] - wav.shape[0])
        noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]
    else:
        start = random.randint(0, wav.shape[0] - noise.shape[0])
        noise_new[start:start + noise.shape[0]] = noise[:]
    wav += noise_new
    return wav

def change_volume(data,p=0.5,db=15):
    gain = random.uniform(-db,db)
    data *=10.**(gain/20.)

def change_speed(data):
    """Change wav speed. can`t larger than 1.1 smaller than 0.9. just use once."""
    fast = 1.05
    slow = 0.95
    old_length = data.shape[0]
    old_indices = np.arange(old_length)
    fast_length = int(old_length / fast)
    slow_length = int(old_length / slow)
    fast_indices = np.linspace(start=0,stop=old_length,num=fast_length)
    slow_indices = np.linspace(start=0,stop=old_length,num=slow_length)
    fast_wav = torch.tensor(np.interp(fast_indices, old_indices, data))
    slow_wav = torch.tensor(np.interp(slow_indices, old_indices, data))
    return fast_wav,slow_wav

def normalize(data):
    mean = np.mean(data, 0, keepdims=True)
    std = np.std(data, 0, keepdims=True)
    data = (data - mean) / (std + 1e-5)
    return data

def fixed_length(audio,len=LENGTH):
    """generate fixed length audio"""
    if audio.size()[0] >= len:
        return audio[0:len]
    else:
        audio = torch.cat((audio,audio),0)
        return fixed_length(audio) 

def cut_save_audio(audio,path,len=4*SAMPLING_RATE):
    idx=0;cnt=0
    while idx+len<audio.shape[0]:
        save_path = path.split('.')[0]+"_"+str(cnt)+".wav"
        torchaudio.save(save_path,audio[idx:idx+len].unsqueeze(-2).float(),SAMPLING_RATE)
        idx=idx+len
        cnt=cnt+1


def path_to_audio(path):
    """Reads and decodes an audio file. data size should have same length"""
    audio,sample_rate=torchaudio.load(path)
    if sample_rate != SAMPLING_RATE:
        print("error samplerate")
    else:
        audio = audio.squeeze()
        return audio

def audio_to_fft(audio):
    """do fft"""
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    # print(audio.size())
    fft = torch.fft.fft(audio)[0:len(audio)//2]
    # plt.plot(fft)
    return torch.abs(fft)

def audio_fft(audio):
    data = audio_to_fft(audio)
    data=data.unsqueeze(dim=-2)
    return data

def audio_melspec(audio,device):
    n_fft = 1000
    win_length = 1000
    hop_length = 400
    n_mels = 200
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    # audio = add_noise(audio,p=0.3)
    mel_spectrogram=mel_spectrogram.to(device)
    melspec = mel_spectrogram(audio.unsqueeze(dim=-2))
    return melspec


class dataset(Dataset):
    """generate dataset and do processing"""
    def __init__(self, audio_paths, labels, option, noisepath,device):
        self.audio_paths = audio_paths
        self.labels = labels
        self.option = option
        self.noisepath=noisepath
        self.device=device

    def __getitem__(self,idx):
        audio_path = self.audio_paths[idx]
        audio = path_to_audio(audio_path)
        audio = fixed_length(audio)
        audio = add_noise(noise_paths=self.noisepath,wav=audio, p=0.3,SNR=15)
        audio = audio.to(self.device)
        if self.option == "FFT":
            data = audio_fft(audio)
        elif self.option == "MelSpec":
             data = audio_melspec(audio,self.device)

        label = torch.tensor(self.labels[idx]).to(self.device)
        return data,label

    def __len__(self):
        return len(self.labels)

def train(dataloader, model, loss_fn, optimizer, device):
    """pytorch train function"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]",flush=True)

def test(dataloader, model, loss_fn, device):
    """pytorch test function"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")