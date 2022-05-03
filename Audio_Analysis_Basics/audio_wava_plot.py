import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

(sample_rate, sig) = wavfile.read(r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t007 t091\clean_data\audio_clips\15.wav")
sample_points = sig.shape[0]
print("sample_rate: %d" % sample_rate)
print("sample_points: %d" % sample_points)
print("time_last = sample_points/sample_rate")

data_0 = sig[:, 0]
time = np.arange(0, sample_points) * (1 / sample_rate)

plt.plot(time, data_0)
plt.show()

k = np.arange(len(data_0))
T = len(data_0)/sample_rate
freq = k/T

DATA_0 = np.fft.fft(data_0)
abs_DATA_0 = abs(DATA_0)

fig = plt.figure('Figure1').add_subplot(111)
fig.plot(freq, abs_DATA_0)
fig.set_xlabel("Frequency/Hz")
fig.set_ylabel("Amplitude")
plt.xlim([0, 1000])

plt.show()
