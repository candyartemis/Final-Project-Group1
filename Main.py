

import numpy as np
import pandas as pd
import librosa
#import IPython.display as ipd
#import math
#import re
#from scipy.io import wavfile
#import matplotlib.pyplot as plt
#from torch.autograd import Variable
import librosa.display
#import os, glob

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim


#----------------------------------------------------------------------------------------------------
# Initial data sizes
#input_size = 131072  #Number of variables per input #using wavfile
input_size = 65536  #Number of variables per input #using librosa
hidden_size = 500   #Number of neurons
num_classes = 8  #8 classes/labels
num_epochs = 25 #Number of iterations
batch_size = 100  #Number of inputs to ran through
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#----------------------------------------------------------------------------------------------------
#Class
classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin']

#----------------------------------------------------------------------------------------------------
#Spit data into test, train, validation sets
print("Loading CSV...")
Medley = pd.read_csv("Medley-solos-DB_metadata.csv")
test = Medley.iloc[0:12236]
test.index = range(len(test.index))

train = Medley.iloc[12236:18077]
train.index = range(len(train.index))

validation = Medley.iloc[18077:]
validation.index = range(len(validation.index))

print("Number of audios=", Medley.shape[0], "  Number of classes=", len(Medley.instrument.unique()))
print(Medley.instrument.unique())


print("Test set size: ")
print(len(test))
print("Train set size: ")
print(len(train))
print("Validation set size: ")
print(len(validation))

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

#----------------------------------------------------------------------------------------------------
# extract class given a known index in csv file
def class_loader(file_index, dataset):
    row = dataset.loc[file_index]
    uuid4_name = str(row.loc['uuid4'])
    classes = str(row.loc['instrument'])
    instrument_id = str(row.loc['instrument_id'])
    return classes, instrument_id, uuid4_name

#----------------------------------------------------------------------------------------------------
# Extract audio features with librosa

def extract_audio(set):
    data = np.zeros((len(set), input_size), dtype=np.float64)

    for i, file in enumerate(set):
        row = set.loc[i]
        uuid4_name = str(row.loc['uuid4'])
        file_name = full_name(uuid4_name)

        #fs, data = wavfile.read(file_name)
        y, sr = librosa.load(file_name)

        data[i,:] = y

        print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

    return data
#print(data)
#----------------------------------------------------------------------------------------------------
#LSTM Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.LSTM(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#----------------------------------------------------------------------------------------------------
#main
model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)





""""
for epoch in range(num_epochs):
    for i, data in enumerate(extract_audio(train)):
        model.zero_grad()
        audios = data
        labels = class_loader(i, train)[1]
        audios = audios.view(batch_size, input_size).cuda()
        audios, labels = Variable(audios), Variable(labels.cuda())


        output = model(audios)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))
                  
#----------------------------------------------------------------------------------------------------
correct = 0
total = 0
for for i, data in enumerate(extract_audio(train)):
    audios = data
    audios = Variable(audios.view(batch_size, input_size)).cuda()
    labels = class_loader(i, train)[1]
    labels = Variable(labels.cuda())
    outputs = model(audios)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    # print(total)
    # print(correct)

print('Accuracy of the network on the 10000 test audio clips: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------

#_, predicted = torch.max(outputs.data, 1)
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in extract_audio(train):
    audios = data
    labels = class_loader(i, train)[1]
    audios = Variable(audios.view(batch_size, input_size)).cuda()
    labels = Variable(labels.cuda())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')
print("--- %s seconds ---" % (time.time() - start_time))
                  
                  
                  
                  
                  
                  
                  """

