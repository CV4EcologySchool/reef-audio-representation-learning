import os
import json
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# for my transformations
#import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, ClippingDistortion, Gain, SevenBandParametricEQ


def resize_mel_spectrogram(mel_spec, desired_shape=(224, 224)):
    # Convert the 2D Mel spectrogram to 4D tensor (batch, channels, height, width)
    mel_spec_tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0)
    # Resize
    resized_mel_spec = F.interpolate(mel_spec_tensor, size=desired_shape, mode='bilinear', align_corners=False)
    return resized_mel_spec.squeeze(0).squeeze(0).numpy()

# augmentation
augment_raw_audio = Compose(
    [
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.0005, p=1), # good
        PitchShift(min_semitones=-2, max_semitones=12, p=0.5), #set values so it doesnt shift too low, rmeoving bomb signal
        TimeStretch(p = 0.5), # defaults are fine
        ClippingDistortion(0, 5, p = 0.5), # tested params to make sure its good
        Gain(-10, 5, p = 0.5), # defaults are fine
        # throws an error, so i commented it out
        #SevenBandParametricEQ(-12, 12, p = 0.5)
    ]
)

# Modify the load_audio_and_get_mel_spectrogram function:
def load_audio_and_get_mel_spectrogram(filename, sr=8000, n_mels=128, n_fft=1024, hop_length=64, win_length=512):
    y, _ = librosa.load(filename, sr=sr)
    augmented_signal = augment_raw_audio(y, sr)

    mel_spectrogram = librosa.feature.melspectrogram(y=augmented_signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_spectrogram_resized = resize_mel_spectrogram(mel_spectrogram)
    return mel_spectrogram_resized



class CTDataset(Dataset):

    def __init__(self, cfg, split, transform):
        '''
            Constructor. Here, we collect and index the dataset inputs and labels.
        '''
        #if split == 'unlabeled':
         #   print('This will not work unless you change the getitem function to have no labels for the unlabeled set') 
        self.data_root = cfg['data_path']
        self.split = split
        self.transform = transform
#

        # index data from JSON file
        self.data = []
        with open(cfg['json_path'], 'r') as f:
            json_data = json.load(f)
            for sublist in json_data.values():
                for entry in sublist:
                    #print(entry)

                    if entry["data_type"] == split:
                        path = entry["file_name"]
                        label = entry["class"]
                        self.data.append([path, label]) ###chNGED TO LIST 

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the audio and get the Mel spectrogram.
        '''
        #print(f'shape of id: {type(idx)}')
        #print(idx)
        audio_path, label = self.data[idx]

        # load audio and get Mel spectrogram
        mel_spectrogram = load_audio_and_get_mel_spectrogram(os.path.join(self.data_root, audio_path))

        # make 3 dimensions, so shape goes from [x, y] to [3, x, y]
        mel_spectrogram_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).repeat(3, 1, 1).float()
        
        # the old transform call, its now ditched
        #if self.transform:
         #   mel_spectrogram_tensor = self.transform(mel_spectrogram_tensor)

        # return the objects, label is commented out for now
        return mel_spectrogram_tensor#, label