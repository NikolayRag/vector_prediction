import numpy as np


def generate(duration=1.0, sampling_rate=500, freqs=[50, 120], amplitudes=[1, 0.5]):
    """
    Generates a sample signal with given frequencies and amplitudes.

    :param duration: Duration of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param freqs: List of frequencies to include in the signal.
    :param amplitudes: List of amplitudes corresponding to the frequencies.
    :return: Tuple of time array and signal array.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for freq, amp in zip(freqs, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    return t, signal


def fft(signal, sampling_rate):
    """
    Performs Fast Fourier Transform (FFT) on a signal.

    :param signal: The input signal.
    :param sampling_rate: Sampling rate of the signal.
    :return: Frequencies and their corresponding magnitudes.
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_magnitude = 2.0 / N * np.abs(fft_result[:N // 2])
    frequencies = np.fft.fftfreq(N, 1 / sampling_rate)[:N // 2]
    
    return frequencies, fft_magnitude


