import numpy as np
import scipy
from scipy.signal import butter, filtfilt, iirnotch

sample_freq = 250

def butterLowpassFilter(data, cutoff_freq, sample_freq, order=5):
    nyquist_freq = 0.5 * sample_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def notchFilter(data, cutoff_freq, sample_freq, Q):
    
    nyquist_freq = 0.5 * sample_freq
    notch_freq = cutoff_freq / nyquist_freq
    b, a = iirnotch(notch_freq, Q)
    y = filtfilt(b, a, data)
    return y


def CarFilter(data):
    channel_average = np.mean(data)
    res = data - channel_average
    return res


def filterData(data) :
    cutoff_lowpass_freq = 5  # Frequência de corte do filtro passa-baixa em Hz
    cutoff_notch_freq = 60 # Frequência de corte do filtro notch
    Q = 30 #Fator de qualidade do filtro Notch
    lowpass_data = butterLowpassFilter(data, cutoff_lowpass_freq, sample_freq)
    notch_data = notchFilter(lowpass_data, cutoff_notch_freq, sample_freq, Q)
    res = CarFilter(notch_data)
    return res

def fftTransform(data):
    fft_res = scipy.fft.fft(data)
    return fft_res
