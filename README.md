# Music Genre Classification
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
```python
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
### 5. Spectral roll-off
```python
import sklearn.preprocessing

# Function to normalize an array
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Load the audio file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

# Compute and norm the spectral roll-off
rolloff_85 = librosa.feature.spectral_rolloff(y=data, sr=sr, roll_percent=0.85)[0]
rolloff_85_norm = normalize(rolloff_85)

# Plot the waveform and normalized spectral roll-offs
plt.figure(figsize=(12, 6))
librosa.display.waveshow(data, sr=sr, alpha=0.4, label='Waveform')
times = librosa.times_like(rolloff_85)
plt.plot(times, rolloff_85_norm, color='r', label='85% roll-off (normalized)')

plt.title('Waveform with Normalized Spectral Roll-offs')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/65fb548f-58e3-4563-9eab-4d4c6973becc)
### 6. Spectral Centroid
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

# Compute the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sr)

# Normalize function
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Plotting
plt.figure(figsize=(10, 4))
librosa.display.waveshow(data, sr=sr, alpha=0.4)
frames = range(len(spectral_centroids[0]))
t = librosa.frames_to_time(frames)
plt.plot(t, normalize(spectral_centroids[0]), color='r', label='Spectral Centroid')

# Set labels and title
plt.title('Waveform and Spectral Centroid over Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/6e684b24-0f98-4049-a767-ceff5a7b09cc)
### 7. Spectral Band Width
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

# Compute the spectral bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sr)

# Plotting
plt.figure(figsize=(10, 4))
librosa.display.waveshow(data, sr=sr, alpha=0.4)
frames = range(len(spectral_bandwidth[0]))
t = librosa.frames_to_time(frames)
plt.plot(t, normalize(spectral_bandwidth[0]), color='r', label='Spectral Bandwidth')

# Set labels and title
plt.title('Waveform and Spectral Bandwidth over Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/c7ff2b75-c56e-46ab-b4ea-0654171ca492)
### 8. Chroma feature
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

chroma = librosa.feature.chroma_stft(y=data,sr=sr)
plt.figure(figsize=(12,6))
librosa.display.specshow(chroma,sr=sr,x_axis="time",y_axis="chroma",cmap="BuPu")
plt.colorbar()
plt.title("Chroma Features")
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/ff4feedb-6658-4625-b730-766da4ed3e9f)
### 9. Harmonic and Percussive
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

# Separate the harmonic and percussive components
data_harmonic, data_percussive = librosa.effects.hpss(data)

# Plot the original, harmonic, and percussive waveforms
plt.figure(figsize=(12, 8))

# Original waveform
plt.subplot(3, 1, 1)
librosa.display.waveshow(data, sr=sr, alpha=0.7)
plt.title('Original Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Harmonic component
plt.subplot(3, 1, 2)
librosa.display.waveshow(data_harmonic, sr=sr, alpha=0.7, color='g')
plt.title('Harmonic Component')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Percussive component
plt.subplot(3, 1, 3)
librosa.display.waveshow(data_percussive, sr=sr, alpha=0.7, color='r')
plt.title('Percussive Component')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/97b53e66-39dc-47e2-a38a-783487798201)
### 10. Mel-Frequency Cepstral Coefficients (MFCC)
```python
# Let load .wav file with default sampling rate of 22,050Hz
data, sr = librosa.load(data_path)

# Compute the MFCCs
mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=21)  # Compute 21 MFCCs to include 0 to 20

# Visualize the MFCCs
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar(label='MFCC Coefficients')
plt.ylabel('MFCC')
plt.title('MFCCs from 0 to 20')
plt.tight_layout()
plt.show()
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/42ef8dfc-dc3a-453d-b5d5-654ba817e69e)

## Building a data pipeline with Dataset and DataLoader
```python
import pandas as pd
# Reading the csv file
df = pd.read_csv("/content/GTZAN/features_3_sec.csv")
df.head()

# Drop the column filename as it is no longer required for training
df=df.drop(labels="filename",axis=1)

# Drop the column length as it is constant
df=df.drop(labels="length",axis=1)

X, y =  df.iloc[:,:-1], df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
# Blues - 0
# Classical - 1
# Country - 2
# Disco - 3
# Hip-hop - 4
# Jazz - 5
# Metal - 6
# Pop - 7
# Reggae - 8
# Rock - 9
encoder=LabelEncoder()
y=encoder.fit_transform(y)

#scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
# splitting 70% data into training set and the remaining 30% to test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=1234)
```
## Feed-forward Neural Network
```python
import os
import numpy as np

import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader, TensorDataset
```
```python
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

input_size = X_train.shape[1]
model = MLP(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000146)
```
## Train the model
```python
num_epochs = 300
batch_size = 256

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

step = 0

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 ==0:
          print(f"Step {step}, Train Loss: {loss.item():.4f}")
        step += 1

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
```
## Evaluate the mode
```python
# Sample testing
model.eval()
with torch.no_grad():
    predictions = model(torch.tensor(X_test).float())
    _, predicted_indices = predictions.max(1)
    print("Expected Index: {}, Predicted Index: {}".format(y_test, predicted_indices.numpy()))

# Confusion Matrix and Classification Report
from sklearn.metrics import classification_report
import seaborn as sns

y_pred = predicted_indices.numpy()
cf_matrix = confusion_matrix(y_test, y_pred)
sns.set(rc={'figure.figsize':(12,6)})
sns.heatmap(cf_matrix, annot=True)
print(classification_report(y_test, y_pred))
```
![image](https://github.com/Buitruongvi/Music-Genre-Classification-Project---btvir/assets/49474873/38a927f7-f42f-42cc-9cc9-278f81a41b02)

## References
AIO2023




