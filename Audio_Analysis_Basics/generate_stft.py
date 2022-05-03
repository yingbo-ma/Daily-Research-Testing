import matplotlib.pyplot as plt
import numpy as np
import librosa.display

audio_path = r"D:\Publications\SIGDIAL_2020\Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\AudioClips\LD2_PKYonge_Class1_Mar142019_B\7\710.wav"

(x, sr) = librosa.load(audio_path)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
print(X.shape)

plt.figure(1)
librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
plt.axis("off")
plt.colorbar(format='%+2.0f dB')
plt.savefig('image.jpg', bbox_inches='tight', pad_inches=0)
plt.show()