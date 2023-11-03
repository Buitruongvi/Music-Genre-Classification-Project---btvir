![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/bc2c493f-deaf-46cd-a4f7-fc99e66a1ad8)# Music Genre Classification
In the field of Music Information Retrieval (MIR), Music Genre Classification plays a crucial role. The [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset, often regarded as the "MNIST of audio," consists of 1,000 music tracks categorized into 10 different genres, from Blues to Rock. Each music track is 30 seconds long and is accompanied by visual representation and extracted feature data.
## Data Download
```python
!gdown 1MGhyeMngD6P9Kz9zJpL68ylQaIQvW7Zx
!unzip GTZAN.zip -d /content/GTZAN
!rm GTZAN.zip
```
## Let check an examples data file
```python
data_path = '/content/GTZAN/genres_original/blues/blues.00000.wav'
import librosa
import matplotlib.pyplot as plt
# Load a .wav file using librosa
data, sr = librosa.load(data_path) # Sampling rate = 22050
# Wave form of the audio
plt.figure(figsize=(12,6))
librosa.display.waveshow(data, color="#2B4F72", alpha = 0.5)
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/30e94410-d022-42cc-8635-c75804bdbf1c)
Although GTZAN has already extracted some features, to gain a deeper understanding of these features, we will start from scratch, directly from the audio files, in order to comprehend the extraction and analysis process of signal features in general and music in particular.
### 1. Spectrogram
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)
# Spectrogram of the audio
stft=librosa.stft(data)
stft_db=librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(12,6))
librosa.display.specshow(stft_db,sr=sr,x_axis='time',y_axis='hz')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/bf59172b-3c54-49e0-9d16-3a15c6c3eb14)
### 2. Mel-Spectrogram
```pyhton
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)
# Creating log mel spectrogram
plt.figure(figsize=(12, 6))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=sr//2)
spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram, y_axis='mel', fmax=sr//2, x_axis='time');
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/5132207d-4dea-4499-9512-9d422c1c4d7c)
### 3. Root mean square Energy (RMS-E)
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)
# Compute RMS
rms = librosa.feature.rms(y=data)
# Plot RMS
plt.figure(figsize=(10, 4))
librosa.display.waveshow(data, sr=sr, alpha=0.4, label='Audio Waveform')
plt.plot(librosa.times_like(rms[0], sr=sr), rms[0], color='r', label='RMS')
plt.legend(loc='upper right')
plt.title('RMS over Time')
plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/6255c6b0-9b99-4cac-85fe-d81039af94b1)
### 4. Zero-crossing rate
```python
## Example of ZCR
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)
# Calculate ZRC of the first 1000 data point of our song
n0 = 0
n1 = 1000
plt.figure(figsize=(14, 5))
plt.plot(data[n0:n1])
plt.grid()
zero_crossings = librosa.zero_crossings(data[n0:n1], pad=False)
print(f'ZRC = {sum(zero_crossings)}')
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/7d01612f-5735-4129-85f2-f9b7dcfaadfc)





