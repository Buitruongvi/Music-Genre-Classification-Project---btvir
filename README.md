# Music Genre Classification
In the field of Music Information Retrieval (MIR), Music Genre Classification plays a crucial role. The [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset, often regarded as the "MNIST of audio," consists of 1,000 music tracks categorized into 10 different genres, from Blues to Rock. Each music track is 30 seconds long and is accompanied by visual representation and extracted feature data.
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


