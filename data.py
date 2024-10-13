import numpy as np
import matplotlib.pyplot as plt
import scipy
from utils import tools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# data chega num array de 8xAmostras
data = np.random.rand(8, 250)
samplingRate = 250
freqs = 4

def mockInputData():
    subject = f"subjects/S{1}.mat"
    data = scipy.io.loadmat(subject)
    return data['data']


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
    res = np.zeros((8, freqs), dtype=object)
    for i in range(8):
        fftData[i] = abs(fftTransform(standardizedData[i,:]))
        res[i] = fftData[i][8], fftData[i][10], fftData[i][12], fftData[i][15]
    ones = np.ones((1,4), dtype=object)
    res = np.concatenate((res, ones), axis=0)
    return res #Nesse momento tenho a FFT da janela inteira


def plotFFTData(fft_res):
    frequencies = [8, 10, 12, 15]
    for j in range(8):
        plt.figure()
        plt.plot(np.abs(fft_res[j,:]))
        for freq in frequencies:
            plt.scatter(freq, np.abs(fft_res[j,freq]), color='red')
            plt.scatter(freq*2, np.abs(fft_res[j,freq*2]), color='red')
            plt.scatter(freq*3, np.abs(fft_res[j,freq*3]), color='red')
            plt.scatter(freq*4, np.abs(fft_res[j,freq*4]), color='red')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(5, 50)
        plt.title(f'FFT Data for Channel {j+1}')
        plt.show()


def plotAllChannelsFFTData(fft_res, i):
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
    plt.title(f'FFT Data for All Channels window {i+1}')
    plt.legend()
    plt.show()


def buildNewLabelMatrix(evokedFreq):
    y = np.ones((9,freqs), dtype=object)*-1
    for i in range(8):
        y[i,evokedFreq-1] = 1
    return y
