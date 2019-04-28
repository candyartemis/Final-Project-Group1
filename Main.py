import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.autograd import Variable
import time
import IPython.display as ipd
import matplotlib.pyplot as plt
import os, glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_time = time.time()
#----------------------------------------------------------------------------------------------------
# Initial data sizes
input_size = 512       #Number of inputs (splited because of the LSTM model)
hidden_size = 500      #Number of neurons
num_classes = 8        #8 classes/labels
num_epochs = 1         #Number of epochs
batch_size = 100       #Number of inputs to ran through
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#----------------------------------------------------------------------------------------------------
#Classes

classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin']

#----------------------------------------------------------------------------------------------------
#Spit data into test, train, validation sets

print("Loading CSV...")
Medley = pd.read_csv("Medley-solos-DB_metadata.csv")

train = Medley.iloc[12236:18077]
train.index = range(len(train.index))

validation = Medley.iloc[18077:]
validation.index = range(len(validation.index))

test = Medley.iloc[0:12236]
test.index = range(len(test.index))

print("Number of audios=", Medley.shape[0], "  Number of classes=", len(Medley.instrument.unique()))
print(Medley.instrument.unique())


print("Train set size: ")
print(len(train))
print("Validation set size: ")
print(len(validation))
print("Test set size: ")
print(len(test))

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
        audio = audio.astype('float')

        sample = {'audio': audio, 'label': instrument_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

audio_dataset_train = MedleyDataset(dataset_csv= train)
audio_dataset_validation = MedleyDataset(dataset_csv= validation)
audio_dataset_test = MedleyDataset(dataset_csv= test)
#----------------------------------------------------------------------------------------------------
#LSTM Model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first = True).cuda()
        self.lstm2tag = nn.Linear(hidden_size, num_classes).cuda()

    def forward(self, x):
        #print(x.shape)
        s = x.shape[0]
        #print(s)
        x = x.reshape(s, 256, 512)
        out, states = self.LSTM(x)
        #print("Afterlstm")
        #print(out.shape)
        #out = out.reshape(batch_size, 256 * hidden_size)
        out = out[:,1,:]
        out = self.lstm2tag(out)
        #print("Afterfinal")
        #print(out.shape)
        return out

#----------------------------------------------------------------------------------------------------
#Train LSTM

model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset = audio_dataset_train, batch_size = batch_size, shuffle = False)
validation_loader = torch.utils.data.DataLoader(dataset = audio_dataset_validation, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = audio_dataset_test, batch_size = batch_size, shuffle = False)


for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        model.zero_grad()

        audios = data['audio']
        labels = data['label']

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

#----------------------------------------------------------------------------------------------------
#Test accuracy on validation set 

correct = 0
total = 0

for i, data in enumerate(validation_loader):
    print(i)
    audios = data['audio']
    labels = data['label']

    audios = audios.type(torch.FloatTensor)
    audios = Variable(audios.cuda())

    outputs = model(audios)

    labels = labels.type(torch.LongTensor)
    labels = Variable(labels.cuda())

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()
    # print(total)
    # print(correct)


print('Accuracy of the network on the 3494 validation audio clips: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
#Test accuracy of each class on Validation set

class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))

for data in validation_loader:
    audios = data['audio']
    labels = data['label']

    audios = audios.type(torch.FloatTensor)
    audios = Variable(audios.cuda())

    outputs = model(audios)

    labels = labels.type(torch.LongTensor)
    labels = Variable(labels.cuda())

    _, predicted = torch.max(outputs.data, 1)

    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------

for i in range(8):
    if (class_total[i] == 0):
        print('Accuracy of %5s : %2d %%' % (classes[i], 0))
    else:
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(model.state_dict(), 'model.pkl')
print("--- %s seconds ---" % (time.time() - start_time))
                  
