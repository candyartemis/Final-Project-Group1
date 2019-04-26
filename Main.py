import numpy as np
import pandas as pd
import librosa
import re
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os, glob
import wave

#----------------------------------------------------------------------------------------------------
#Spit data into test, train, validation sets
print("Loading CSV...")
Medley = pd.read_csv("Medley-solos-DB_metadata.csv")
test = Medley.iloc[0:12236]
train = Medley.iloc[12236:18077]
validation = Medley.iloc[18077:]

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
# Extract audio features with librosa
def extract_audio_features(file):
    "Extract audio features from an audio file for class classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

#---------------------------------------------------------------------------------------------------
# extract all audio features given a known index in csv file
def data_loader(file_index):
    row = Medley.loc[file_index].iloc[0]
    uuid4_name = str(row.loc['uuid4'])
    file_name = full_name(uuid4_name)

    #fs, data = wavfile.read(file_name)
    #print("Length of current audio: ")
    #print(len(data))
    #plt.plot(data)
    # plt.show()

    libros_features = extract_audio_features(file_name)

    return libros_features
#----------------------------------------------------------------------------------------------------
#Class tagging

classes = ['clarinet', 'distorted electric guitar', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin']

def get_class(model, audio_path):
    "Predict genre of music using a trained model"
    prediction = model.predict(extract_audio_features(audio_path))
    predict_class = classes[np.argmax(prediction)]
    return predict_class

#----------------------------------------------------------------------------------------------------
#LSTM Model







#---------------------------------------------------------------------------------------------------
#write a loop toget each uuid4 in test/train/validation, f
# ull_name it and get the data, then extract audio features
# use the return value to do LSTM
#index = Medley.index
#print(index)
#for i in enumerate(index):
    #lib_feature = data_loader(i)


