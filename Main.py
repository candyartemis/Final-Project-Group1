import numpy as np
import pandas as pd
import librosa
#import IPython.display as ipd
#import math
#import re
from scipy.io import wavfile
#import matplotlib.pyplot as plt
from torch.autograd import Variable
import librosa.display
import os, glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#----------------------------------------------------------------------------------------------------
# Initial data sizes
input_size = 131072  #Number of variables per input #using wavfile
#input_size = 65536  #Number of variables per input #using librosa
#input_size = 13  #Number of input @using mfcc
hidden_size = 500   #Number of neurons
num_classes = 8  #8 classes/labels
num_epochs = 25 #Number of iterations
batch_size = 100  #Number of inputs to ran through
learning_rate = 0.001
embed_size = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#----------------------------------------------------------------------------------------------------
#Class
classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin']

#----------------------------------------------------------------------------------------------------
#Spit data into test, train, validation sets
print("Loading CSV...")
Medley = pd.read_csv("Medley-solos-DB_metadata.csv")
#test = pd.read_csv("Medley_audio_test.csv")
#train = pd.read_csv("Medley_audio_train.csv")
#test = Medley.iloc[0:12236]
#test.index = range(len(test.index))

train = Medley.iloc[12236:18077]
train.index = range(len(train.index))

#validation = Medley.iloc[18077:]
#validation.index = range(len(validation.index))

print("Number of audios=", Medley.shape[0], "  Number of classes=", len(Medley.instrument.unique()))
print(Medley.instrument.unique())


#print("Test set size: ")
#print(len(test))
print("Train set size: ")
print(len(train))
#print("Validation set size: ")
#print(len(validation))

#---------------------------------------------------------------------------------------------------
#Get file uuid4
filenames = Medley.uuid4
print(filenames.head())
Medley['audio'] = ""
Medley.head()

#---------------------------------------------------------------------------------------------------
#Load audio file linked to the uuid
def full_name(file):
    correspding_row = Medley.loc[Medley['uuid4'] == file].iloc[0]
    subset = str(correspding_row.loc['subset'])
    instrument_id = str(correspding_row.loc['instrument_id'])
    parts = ['Medley-solos-DB_', str(subset), '-', str(instrument_id), '_', file, '.wav.wav']
    s = ''
    file_name = s.join(parts)
    return file_name

def load_file(file):
    file_name = full_name(file)
    path = '/home/ubuntu/Machine-Learning/Medley-solos-DB/'
    parts = [path, file_name]
    s = ''
    link = s.join(parts)
    return link
#----------------------------------------------------------------------------------------------------
# extract class given a known index in csv file
def class_loader(file_index, dataset):
    row = dataset.loc[file_index]
    uuid4_name = str(row.loc['uuid4'])
    classes = str(row.loc['instrument'])
    instrument_id = str(row.loc['instrument_id'])
    return classes, instrument_id, uuid4_name

#---------------------------------------------------------------------------------------------------
#Process Dataset

class MedleyDataset(Dataset):
    def __init__(self, dataset_csv, transform=None):
        self.dataset_frame = dataset_csv
        self.transform = transform

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, index):
        uuid4 = self.dataset_frame.iloc[index, 4]
        instrument_list = self.dataset_frame.iloc[index, 2]
        instrument_id = int(instrument_list)
        link = load_file(uuid4)
        fs, audio = wavfile.read(link)
        #y, sr = librosa.load(link)
        #mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        #print(audio)
        #audio = mfcc.T[0: 128, :]
        audio = audio.astype('float')

        sample = {'audio': audio, 'label': instrument_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

audio_dataset_train = MedleyDataset(dataset_csv= train)
#----------------------------------------------------------------------------------------------------
#LSTM Model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.Linear = nn.Linear(input_size, hidden_size).cuda()
        self.relu = nn.ReLU().cuda()
        self.fc2 = nn.Linear(hidden_size, num_classes).cuda()

    def forward(self, x):
        print(x.shape)
        out = self.Linear(x)
        #print("Afterlstm")
        #print(out.shape)
        out = self.relu(out)
        out = self.fc2(out.view(len(x), -1))
        return out

#----------------------------------------------------------------------------------------------------
#Train LSTM

model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset = audio_dataset_train, batch_size = batch_size, shuffle = False)

#print(train_loader)

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        model.zero_grad()

        audios = data['audio']
        labels = data['label']
        print(i)

        audios = audios.type(torch.FloatTensor)
        audios = Variable(audios.cuda())

        output = model(audios)

        labels = labels.type(torch.LongTensor)
        labels = Variable(labels.cuda())

        loss = criterion(output, labels).cuda()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))
