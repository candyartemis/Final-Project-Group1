#%%

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import wave
import os, glob
from scipy.io import wavfile
import librosa
import csv

#%%
#Read in metadata
Medley = pd.read_csv('Medley-solos-DB_metadata.csv')
Medley.head()

#%%
#Metadata
print("Number of examples=", Medley.shape[0], "Number of classes=", len(Medley.instrument.unique()))
print()
print("Class names: ", Medley.instrument.unique())
print()

#%%
#Split into train, validation, and test sets

train_df = Medley.loc[Medley["subset"] == "train"]
validation_df = Medley.loc[Medley["subset"] == "validation"]
test_df = Medley.loc[Medley["subset"] == "test"]

print(train_df.head())
print()
print(validation_df.head())
print()
print(test_df.head())

#%%
#Function to get full name of audio files
def full_name(file):
    correspding_row = Medley.loc[Medley['uuid4'] == file].iloc[0]
    subset = str(correspding_row.loc['subset'])
    instrument_id = str(correspding_row.loc['instrument_id'])
    parts = ['Medley/Medley-solos-DB/','Medley-solos-DB_', str(subset), '-', str(instrument_id), '_', file, '.wav.wav']
    s = ''
    file_name = s.join(parts)
    return file_name

#%%
#EX
file_name = full_name('fc0c75c5-f324-5133-fc41-2492f03991c6')
fs, data = wavfile.read(file_name)

print("Length of current audio: ", len(data))
plt.plot(data)
plt.show()

#%%
filenames = Medley.uuid4
with open('Medley_audio.csv','a') as fd:
    writer = csv.writer(fd)
    for file in filenames:
        file_name = full_name(file)
        fs, data = wavfile.read(file_name)
        label = Medley.loc[Medley['uuid4'] == file, 'instrument_id'].iloc[0]
        subset =  Medley.loc[Medley['uuid4'] == file, 'subset'].iloc[0]
        fields = [subset, label, data]
        writer.writerow(fields)
fd.close()
filenames

#%%
Medley_audio = pd.read_csv('Medley_audio.csv')
Medley_audio.head()
Medley_audio[1,1]
#%%
train_set = torchvision.datasets.FashionMNIST(root='./data_fashion', train=True, download=True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data_fashion', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#%%
for file in filenames:
    file_name = full_name(file)
    fs, data = wavfile.read(file_name)
    label = Medley.loc[Medley['uuid4'] == file, 'instrument_id'].iloc[0]
    subset =  Medley.loc[Medley['uuid4'] == file, 'subset'].iloc[0]
    fields = [subset, label, data]
    writer.writerow(fields)
