#The commented part is the same as the music_feature_loader
#Need to run music_feature_loader first, and have the 3 .npy file, to run this code.




import numpy as np
import pandas as pd
from torch.autograd import Variable
import time
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------
# Initial data sizes
input_size = 63  # Number of inputs (splited because of the LSTM model) using Librosa
hidden_size1 = 500  # Number of neurons
hidden_size2 = 30  # Number of neurons
num_classes = 8  # 8 classes/labels
num_epochs = 300  # Number of epochs
batch_size = 100  # Number of audio clips to ran through 1 iteration
learning_rate = 0.001
confusion_m = np.zeros((8, 8))
# ----------------------------------------------------------------------------------------------------
# Classes

classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet',
           'violin']

# ----------------------------------------------------------------------------------------------------
# Spit data into test, train, validation sets

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

"""""
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

def find_file(file):
    file_name = full_name(file)
    path = '/home/ubuntu/Machine-Learning/Medley-solos-DB/'
    parts = [path, file_name]
    s = ''
    link = s.join(parts)
    return link

#---------------------------------------------------------------------------------------------------
#Generating music data ndarray in frequency domain using librosa

def get_music_features(dataset):
    timeseries_length = 3
    audio = np.zeros((len(dataset), timeseries_length, 21), dtype=np.float64)

    for i in range(len(dataset)):
        row = dataset.loc[i]
        uuid4_name = str(row.loc['uuid4'])
        link = find_file(uuid4_name)

        y, sr = librosa.load(link)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        audio[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
        audio[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
        audio[i, :, 14:21] = spectral_contrast.T[0:timeseries_length, :]
        

        if ((i + 1) % 100 == 0):
            print("Extracted features audio clip %i of %i." % (i + 1, len(dataset)))

    return audio

train_audio_data = get_music_features(train)
validation_audio_data = get_music_features(validation)
test_audio_data = get_music_features(test)

np.save('train_audio_data.npy', train_audio_data)
np.save('validation_audio_data.npy', validation_audio_data)
np.save('test_audio_data.npy', test_audio_data)
"""
# ---------------------------------------------------------------------------------------------------
# load npy music feature matrix data
train_audio_data = np.load('train_audio_data.npy')
validation_audio_data = np.load('validation_audio_data.npy')
test_audio_data = np.load('test_audio_data.npy')


# ---------------------------------------------------------------------------------------------------
# Process Dataset

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
        set = self.dataset_frame
        if (set.shape == train.shape):
            feature_matrix = train_audio_data
        if (set.shape == validation.shape):
            feature_matrix = validation_audio_data
        if (set.shape == test.shape):
            feature_matrix = test_audio_data

        audio = feature_matrix[index, :, :]

        audio = audio.astype('float')

        sample = {'audio': audio, 'label': instrument_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


audio_dataset_train = MedleyDataset(dataset_csv=train)
audio_dataset_validation = MedleyDataset(dataset_csv=validation)
audio_dataset_test = MedleyDataset(dataset_csv=test)


# ----------------------------------------------------------------------------------------------------
# LSTM Model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.LSTM1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.1).cuda()
        #self.LSTM2 = nn.LSTM(hidden_size1, hidden_size2, batch_first = True).cuda()
        self.lstm2tag = nn.Linear(hidden_size1, num_classes).cuda()

    def forward(self, x):
        #print(x.shape)
        s = x.shape[0]
        x = x.reshape(s, 1, input_size)
        out, states = self.LSTM1(x)
        #out, states = self.LSTM2(out)
        out = out[:, 0, :]
        out = self.lstm2tag(out)
        return out


# ----------------------------------------------------------------------------------------------------
# Train LSTM

model = Net(input_size, hidden_size1, hidden_size2, num_classes)
criterion = nn.CrossEntropyLoss()


#optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.8, eps = 1e-6, lr=learning_rate)
#optimizer = torch.optim.SGD(params = model.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)

train_loader = torch.utils.data.DataLoader(dataset=audio_dataset_train, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=audio_dataset_validation, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = audio_dataset_test, batch_size = batch_size, shuffle = True)


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
        optimizer.zero_grad()


        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))



fig, ax = plt.subplots()
ax.plot(epochs, loss_index)

ax.set(xlabel='Epoch', ylabel='Performance Index', title='Performance Index Over Time')
ax.grid()
fig.savefig("test.png")
plt.show()


# ----------------------------------------------------------------------------------------------------
# Test accuracy on validation set

correct = 0
total = 0

for i, data in enumerate(validation_loader):

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


    leng = predicted.shape[0]

    a = predicted.cpu().detach().numpy()
    b = labels.cpu().detach().numpy()


    for j in range(leng):
        k1 = a[j]
        k2 = b[j]
        confusion_m[k1,k2] += 1

#confusion matrix
confusion_m.astype(int)
ax = sns.heatmap(confusion_m, annot=True)
print(confusion_m)
print('Accuracy of the network on the 3494 validation audio clips: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
# Test accuracy of each class on Validation set

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
    s = labels.shape[0]
    for i in range(s):
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
torch.save(model.state_dict(), 'model_time.pkl')

# ----------------------------------------------------------------------------------------------------
# Test accuracy on test set

correct = 0
total = 0

for i, data in enumerate(test_loader):

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


    leng = predicted.shape[0]

    a = predicted.cpu().detach().numpy()
    b = labels.cpu().detach().numpy()


    for j in range(leng):
        k1 = a[j]
        k2 = b[j]
        confusion_m[k1,k2] += 1

#confusion matrix
confusion_m.astype(int)
ax = sns.heatmap(confusion_m, annot=True)
print(confusion_m)
print('Accuracy of the network on the 3494 validation audio clips: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
# Test accuracy of each class on test set

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

# --------------------------------------------------------------------------------------------

for i in range(8):
    if (class_total[i] == 0):
        print('Accuracy of %5s : %2d %%' % (classes[i], 0))
    else:
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(model.state_dict(), 'model_time.pkl')
