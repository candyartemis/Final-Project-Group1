# Final-Project-Group1-Audio Tagging with LSTM

We are using the [Medley-solos](https://zenodo.org/record/2582103#.XMOHKi2ZNE5) dataset for musical instrument recognision. Each audio clip in training set and test set is 3-second long, and contains a single instrument sound. There are 8 classes of instruments in total: clarinet, distorted electric guitar, female singer, flute, piano, tenor saxophone, trumpet, and violin.

LTSM model is selected for the audio tagging since it is a time series problem. We extract the time domain data from the audio clips with wavfile, and time series data (MFCC, spectral centroid and spectral contrast) with librosa, and write the code using Pytorch framework.

# Twofold Analysis
* Main(time).py uses time domain data extracted from wavfile.
* Main(frequency).py uses time domain data extracted from librosa.
* music_feature_loader.py needs to run first to created the 3 .npy files that will be used in main(frequency).py

### Prerequisites

To test the code, install librosa
```
conda install -c conda-forge librosa
```
or
```
pip install librosa
```
note that if you install via pip on a ubuntu based system, libav-tools is also needed for loading audio files
```
sudo apt_get install libav-tools
```
## Uses
* Plain music data from time domain (generated with wavfile)
* MFCC
* Spectral centroid
* Spectral contrast

## Built With

* [Pytorch](https://pytorch.org) - The framework used
* [numpy](https://maven.apache.org/) - Dependency used
* [Librosa](https://www.numpy.org) - Used to extract features


## Authors

* **Madison Turano** 
* **Jingjing Xu**


## Acknowledgments

* Thanks to [ruohoruotsi](https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification)'s code on Music Genre Classification with LSTMs
