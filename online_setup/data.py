from utils import  tools
import numpy as np

# data chega num array de 8xAmostras
data = np.random.rand(8, 2500) * 100
samplingRate = 250

def carFilter(data):
    mean = np.mean(data, axis=1)
    car_data = data - mean.reshape(-1,1)
    return car_data

def standardizeData(data):
    cutoff_lowpass_freq = 5
    cutoff_notch_freq = 60
    Q = 30 
    lowpass_data = tools.butterLowpassFilter(data, cutoff_lowpass_freq, samplingRate)
    notch_data = tools.notchFilter(lowpass_data, cutoff_notch_freq, samplingRate, Q)
    res = carFilter(notch_data)
    return res

def buildWindow(data, samplingRate, windowSize):
    data = standardizeData(data)
    numberOfWindows = len(data[0])/(samplingRate*windowSize)
    windowData = np.zeros((8, numberOfWindows))
    for j in range(8):
        for i in range(numberOfWindows):
            windowData[j,i] = data[j,i*samplingRate*windowSize:(i+1)*samplingRate*windowSize]
    return windowData

def fftTransform(data):
    fft_res = np.fft.fft(data)
    return fft_res
