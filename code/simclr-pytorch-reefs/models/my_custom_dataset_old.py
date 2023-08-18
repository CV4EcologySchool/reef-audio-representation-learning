'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

""""
    Turan adjusted this to work on his ROV data. Ben now adjusting to work on reef audio.
    This will make a custom dataset.

"""

### turans old imports
# import os
# import json
# from torch.utils.data import Dataset
# from torchvision.transforms import Compose, Resize, ToTensor
# from PIL import Image
# import csv
# from torchvision.transforms import transforms
# from torchvision import transforms, datasets
################


# Preprocessor classes are used to load, transform, and augment audio samples for use in a machine learing model
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset


# helper function for displaying a sample as an image
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid

#other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess

# Imports by ben
import json

# path to data
dataset_path = r'/home/ben/data/full_dataset/'
json_path = r'/home/ben/data/dataset.json'

# what does this do 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Load the JSON data from the file
with open(json_path, "r") as file:
    data = json.load(file)

# Convert the list of dictionaries (which is the value of the main dictionary) into a DataFrame
df = pd.DataFrame(data[list(data.keys())[0]])

# Create a dataframe with just file_path and a class column (req for AudioFileDataset)
transformed_df = df[['file_name', 'class']].copy()

# rename 'file_name' column to 'file'
transformed_df.rename(columns={'file_name': 'file'}, inplace=True)

# set file to be the index for AudioFileDataset
transformed_df.set_index('file', inplace=True)

# set all classes to 1 as AudioFileDataset requires class
transformed_df['class'] = 1

# append dataset_path to start of file_name column
transformed_df.index = dataset_path + transformed_df.index
#transformed_df.head() # for notebook

# initialize the preprocessor (forget what this does?)
pre = SpectrogramPreprocessor(sample_duration=1.92)

# initialize the dataset
dataset = AudioFileDataset(transformed_df, pre)

# change the bandpass from the default to 8khz
dataset.preprocessor.pipeline.bandpass.set(min_f=0,max_f=8000)



# for redefining the actions in the dataset.preprocessor.pipeline
from opensoundscape import Action
from opensoundscape.spectrogram import MelSpectrogram


# custom functions to produce melspetrograms
def melspec_linear_to_db(melspec):
    
    # because there's an underflow error during MelSpectrogram.from_audio() with dB_scale = True,
    # we instead perform dB scaling afterwards
    # which for some mysterious reason works
    
    melspectrogram = 10 * np.log10(
                    melspec.spectrogram,
                    where=melspec.spectrogram > 0,
                    out=np.full(melspec.spectrogram.shape, -np.inf),)

    # limit the decibel range (-100 to -20 dB by default)
    # values below lower limit set to lower limit,
    # values above upper limit set to uper limit
    min_db, max_db = melspec.decibel_limits
    melspectrogram[melspectrogram > max_db] = max_db
    melspectrogram[melspectrogram < min_db] = min_db

    return MelSpectrogram(times=melspec.times,
                          frequencies=melspec.frequencies,
                          spectrogram=melspectrogram,
                          decibel_limits=melspec.decibel_limits,                   
    )

def my_melspec(audio):
    melspec_linear = MelSpectrogram.from_audio(audio,dB_scale=False, window_samples = 512) #adjust params, use MelSpectrogram.from_audio to see what these are
    melspec_db = melspec_linear_to_db(melspec_linear)
    return melspec_db

# define the melspec action to be used in the pipeline, using functions above
melspec_action = Action(my_melspec)

# redfine the bandpass action to work on melspecs
melspec_bandpass_action = Action(MelSpectrogram.bandpass, min_f = 0, max_f = 8000)

# redefine the spectrogram function to melspec using the functions above
dataset.preprocessor.pipeline['to_spec'] = melspec_action

# Change to bandpass melspecs not normal specs
dataset.preprocessor.pipeline['bandpass'] = melspec_bandpass_action

# Check shape of each sample
print(f"Data shape is: {dataset[0].data.shape}")

print('Succesfully run')

 