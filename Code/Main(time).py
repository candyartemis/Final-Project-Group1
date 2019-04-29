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
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

#%%
# Initial data sizes
input_size = 131072       #Number of inputs (splited because of the LSTM model)
hidden_size1 = 500      #Number of neurons
hidden_size2 = 30     #Number of neurons
num_classes = 8        #8 classes/labels
num_epochs = 30     #Number of epochs
batch_size = 100       #Number of audio clips to ran through 1 iteration
learning_rate = 0.001
confusion_m = np.zeros((8,8))

#%%
#Classes

classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 
           'piano', 'tenor saxophone', 'trumpet', 'violin']

#%%
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
print()
print(Medley.instrument.unique())


print("Train set size: ", len(train))
print("Validation set size: ", len(validation))
print("Test set size: ", len(test))

#%%
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
    path = '/home/ubuntu/Final-Project-Group1/Medley-solos-DB/'
    parts = [path, file_name]
    s = ''
    link = s.join(parts)
    return link

#%%

#%%
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
#%%
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
#%%
                                
audio_dataset_train = MedleyDataset(dataset_csv= train)
audio_dataset_validation = MedleyDataset(dataset_csv= validation)
audio_dataset_test = MedleyDataset(dataset_csv= test)

#%%
#LSTM Model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.LSTM1 = nn.LSTM(input_size, hidden_size1, batch_first = True, num_layers = 2, dropout = 0.1).cuda()
        self.LSTM2 = nn.LSTM(hidden_size1, hidden_size2, batch_first = True).cuda()
        self.lstm2tag = nn.Linear(hidden_size2, num_classes).cuda()

    def forward(self, x):
        #print(x.shape)
        s = x.shape[0]
        #print(s)
        x = x.reshape(s, 1, input_size)
        out, states = self.LSTM1(x).cuda()
        out, states = self.LSTM2(out).cuda()
        #print("Afterlstm")
        #print(out.shape)
        out = out[:,0,:]
        out = self.lstm2tag(out)
        return out

#%%
#Train LSTM

model = Net(input_size, hidden_size1, hidden_size2, num_classes)

#%%
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)
#optimizer = torch.optim.SGD(params = model.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = torch.optim.Adam(params = model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

train_loader = torch.utils.data.DataLoader(dataset = audio_dataset_train, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = audio_dataset_validation, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = audio_dataset_test, batch_size = batch_size, shuffle = False)

#%%
start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = np.array([])
loss_index = np.array([])

for epoch in range(num_epochs):
    for i, data in enumerate(test_loader):

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
        optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(test) // batch_size, loss.data[0]))
    
    epochs = np.append(epochs, epoch)
    loss_index = np.append(loss_index, loss.item())

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)

#%%

plt.figure(figsize = (12,8))
plt.plot(epochs, loss_index, 'r')
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("Performance Index", fontsize = 14)
plt.title("Performance Index Over Time for Training Set", fontsize = 20)
plt.show()

#%%
#Test accuracy on validation set

correct = 0
total = 0

epochs_train = np.array([])
loss_index_train = np.array([])

for i, data in enumerate(train_loader):
    audios = data['audio']
    labels = data['label']

    audios = audios.type(torch.FloatTensor)
    audios = Variable(audios.cuda())

    outputs = model(audios)

    labels = labels.type(torch.LongTensor)
    labels = Variable(labels.cuda())

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    #correct += (predicted == labels).sum()
    for j in range(len(predicted)):
        if predicted[j] == labels[j]: correct += 1
    # print(total)
    # print(correct)
    epochs = np.append(epochs_train, epoch)
    loss_index = np.append(loss_index_train, loss.item())


#Confusion matrix
results = confusion_matrix(labels, predicted)
confusion_m = confusion_m + results

print(confusion_m)
print('Accuracy of the network on the 3494 validation audio clips: %d %%' % (100 * correct / total))

#%%
plt.figure(figsize = (12,8))
plt.plot(epochs_test, loss_index_test, 'r')
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("Performance Index", fontsize = 14)
plt.title("Performance Index Over Time for Test Set", fontsize = 20)
plt.show()

#%%
#Test accuracy of each class on Validation set

class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))

for data in test_loader:
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
    s = labels.shape[0]
    for i in range(s):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

#%%

for i in range(8):
    if (class_total[i] == 0):
        print('Accuracy of %5s : %2d %%' % (classes[i], 0))
    else:
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

#%%
torch.save(model.state_dict(), 'model_time.pkl')
