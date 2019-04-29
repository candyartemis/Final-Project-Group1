#This file extracts the music feature (mfcc, spectral centroid, spectral contrast) using librosa, and save it as.npy numpy ndarray 
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
#Generating music data matrix in frequency domain using librosa

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
