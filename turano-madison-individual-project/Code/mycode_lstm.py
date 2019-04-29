# -*- coding: utf-8 -*-
"""
I wrote this code as a first draft for processing data and training our model.

"""
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
input_size = 131072  #Number of variables per input
hidden_size = 500   #Number of neurons
num_classes = 8  #10 classes/labels
num_epochs = 25 #Number of iterations
batch_size = 100  #Number of inputs to ran through
learning_rate = 0.001

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
#Function to get full name of audio files
def full_name(file):
    correspding_row = Medley.loc[Medley['uuid4'] == file].iloc[0]
    subset = str(correspding_row.loc['subset'])
    instrument_id = str(correspding_row.loc['instrument_id'])
    parts = ['Medley-solos-DB/','Medley-solos-DB_', str(subset), '-', str(instrument_id), '_', file, '.wav.wav']
    s = ''
    file_name = s.join(parts)
    return file_name

#%%
#Example
file_name = full_name('fc0c75c5-f324-5133-fc41-2492f03991c6')
fs, data = wavfile.read(file_name)

print("Length of current audio: ", len(data))
plt.plot(data)
plt.show()

#%%
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

# In[60]:

net = Net(input_size, hidden_size, num_classes)
#net.cuda()

# In[61]:

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adadelta(net.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)

# In[61]:

train_df = Medley.loc[Medley["subset"] == "train"]
validation_df = Medley.loc[Medley["subset"] == "validation"]
test_df = Medley.loc[Medley["subset"] == "test"]

print(train_df.head())
print()
print(validation_df.head())
print()
print(test_df.head())

#%%
filenames = Medley.uuid4
train_input = np.array([])
train_target = np.array([])
validation_input = np.array([])
validation_target = np.array([])
test_input = np.array([])
test_target = np.array([])

for file in filenames:
    file_name = full_name(file)
    fs, data = wavfile.read(file_name)
    label = Medley.loc[Medley['uuid4'] == file, 'instrument_id'].iloc[0]
    subset =  Medley.loc[Medley['uuid4'] == file, 'subset'].iloc[0]
    if subset == 'train':
        train_input = np.append(train_input, data)
        train_target.append(label)
    elif subset == 'validation':
        validation_input = np.append(validation_input, data)
        validation_target = np.append(validation_target, label)
    else:
        test_input = np.append(test_input, data)
        test_target = np.append(test_target, label)

# In[61]:
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
dtype = torch.float

for epoch in range(num_epochs):  
    for i in range(len(train_input)):
        #Get data
        data = torch.from_numpy(train_input[i], dtype = dtype).to(device)
        label = torch.from_numpy(train_target[i], dtype = dtype).to(device)
            
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                %(epoch+1, num_epochs, i+1, len(train_df)//batch_size, loss.item()))

#%%
correct = 0
total = 0
for i in range(len(test_input)):
    data = torch.from_numpy(test_input[i], dtype = dtype).to(device)
    outputs = net(data)
    label = torch.from_numpy(test_target[i], dtype = dtype).to(device)
    label = torch.tensor(label, dtype=torch.long, device=device)
    predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    for i in range(len(predicted[1])):
        if predicted[1][i] == labels[i]: correct += 1

print('Accuracy of the network on the 10000 test images: %d%%' % (100 * correct / total))

#%%
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#%%


class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
for i in range(len(test_input)):
    data = torch.from_numpy(test_input[i], dtype = dtype).to(device)
    outputs = net(data)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_target[i]
    c = (predicted.cpu().numpy() == labels)
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
#%%


for i in range(8):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


#%%


torch.save(net.state_dict(), 'model.pk1')
