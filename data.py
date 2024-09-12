import numpy as np
import matplotlib.pyplot as plt
from utils import tools

# Não preciso janelar, janelas 

# data chega num array de 8xAmostras
data = np.random.rand(8, 250)
samplingRate = 250

def carFilter(data):
    mean = np.mean(data, axis=0)
    car_data = data - mean.reshape(1,-1)
    return car_data

def standardizeData(data):
    cutoff_lowpass_freq = 5
    cutoff_notch_freq = 60
    Q = 30 
    lowpass_data = tools.butterLowpassFilter(data, cutoff_lowpass_freq, samplingRate)
    notch_data = tools.notchFilter(lowpass_data, cutoff_notch_freq, samplingRate, Q)
    res = carFilter(notch_data)
    return res

# ------------------------------------------------------------------------------------------------------------------------------------------
def fftTransform(windowData):
    fft_res = np.fft.fft(windowData)
    return fft_res

def buildFFTData(data):
    standardizedData = standardizeData(data)
    fftData = np.zeros((8, len(standardizedData[0])), dtype=object)
    for i in range(8):
        fftData[i] = abs(fftTransform(standardizedData[i,:]))
    return fftData

def getMaxOutFrequency(fft_res):
    maxFreqMatrix = np.zeros((8, len(fft_res[0])), dtype=object)
    for j in range(8):
        for i in range(len(fft_res[0])):
            maxFreqMatrix[j,i] = [fft_res[j,i][8], fft_res[j,i][10], fft_res[j,i][12], fft_res[j,i][15]]
    return maxFreqMatrix

def mainDataPreparation(data):
    windowData = buildFFTData(data)
    fftData = np.zeros((8, len(windowData[0])), dtype=object)
    for j in range(8):
        for i in range(len(windowData[0])):
            fftData[j,i] = fftTransform(windowData[j,i])
    maxOut = getMaxOutFrequency(fftData)
    return maxOut


def DataMain(data):
    ans = mainDataPreparation(data)



def plotFFTData(fft_res):
    frequencies = [8, 10, 12, 15]
    for j in range(8):
        plt.figure()
        plt.plot(np.abs(fft_res[j,:]))
        for freq in frequencies:
            plt.scatter(freq, np.abs(fft_res[j,freq]), color='red')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(5, 50)
        plt.title(f'FFT Data for Channel {j+1}')
        plt.show()


def plotAllChannelsFFTData(fft_res):
    frequencies = [8, 10, 12, 15]
    plt.figure()
    for j in range(8):
        plt.plot(np.abs(fft_res[j,:]), label=f'Channel {j+1}')
        for freq in frequencies:
            plt.scatter(freq, np.abs(fft_res[j,freq]), color='red')
            plt.scatter(freq*2, np.abs(fft_res[j,freq*2]), color='red')
            plt.scatter(freq*3, np.abs(fft_res[j,freq*3]), color='red')
            plt.scatter(freq*4, np.abs(fft_res[j,freq*4]), color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(5, 50)
    plt.title('FFT Data for All Channels')
    plt.legend()
    plt.show()

plotAllChannelsFFTData(buildFFTData(data))