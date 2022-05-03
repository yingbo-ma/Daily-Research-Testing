from librosa.feature import mfcc
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

audio_path = "/home/yingbo/FLECKS/FlecksCode/python/AudioSignalAnalysis/n_2.wav"
(x, sr) = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()