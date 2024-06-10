'''
Test: calculate data's FFT
'''


import _fixpath
from _jsonadd import *
from _plotlib import *

import mtools


sampling_rate = 500  # Hz

# Generate an example signal
data = load_json()
if data is None:
    duration = 2.0  # seconds
    freqs = [50, 120]  # Hz
    amplitudes = [1.0, 0.5]

    t, data = mtools.generate(duration, sampling_rate, freqs, amplitudes)
else:
    data = data[:,1]

# Perform FFT
frequencies, fft_magnitude = mtools.fft(data, sampling_rate)
jdata = {}
jdata["low"] = frequencies[1]
jdata["high"] = frequencies[-1]
jdata["data"] = fft_magnitude

# Plot the data and its frequency spectrum
plot_start()
#plot_vectors_layer(pd.DataFrame(data))
plot_vectors_layer(pd.DataFrame(fft_magnitude[2:]))
plot_end()

print(jdata)
