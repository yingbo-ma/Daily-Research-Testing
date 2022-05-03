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

from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def frequency_sepectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    print(n)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


# Sine sample with a frequency of 1hz and add some noise
sr = 32  # sampling rate
y = np.linspace(0, 2*np.pi, sr)
y = np.tile(np.sin(y), 5)
y += np.random.normal(0, 1, y.shape)
t = np.arange(len(y)) / float(sr)

plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')

frq, X = frequency_sepectrum(y, sr)

plt.subplot(2, 1, 2)
plt.plot(frq, X, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.tight_layout()


# wav sample from https://freewavesamples.com/files/Alesis-Sanctuary-QCard-Crickets.wav
here_path = os.path.dirname(os.path.realpath(__file__))
wav_file_name = r'E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t007 t091\clean_data\audio_clips\0.wav'
sr, signal = wavfile.read(wav_file_name)

y = signal[:, 0]  # use the first channel (or take their average, alternatively)
t = np.arange(len(y)) / float(sr)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')

frq, X = frequency_sepectrum(y, sr)

plt.subplot(2, 1, 2)
plt.plot(frq, X, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.tight_layout()

plt.show()
