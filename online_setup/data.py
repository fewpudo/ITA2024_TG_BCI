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

# ------------------------------------------------------------------------------------------------------------------------------------------

def buildWindow(data, samplingRate, windowSize):
    data = standardizeData(data)
    numberOfWindows = len(data[0])/(samplingRate*windowSize)
    windowData = np.zeros((8, numberOfWindows))
    for j in range(8):
        for i in range(numberOfWindows):
            windowData[j,i] = data[j,i*samplingRate*windowSize:(i+1)*samplingRate*windowSize]
    return windowData

def fftTransform(windowData):
    fft_res = np.fft.fft(windowData)
    return fft_res

def getMaxOutFrequency(fft_res):
    maxFreqMatrix = np.zeros((8, len(fft_res[0])), dtype=object)
    for j in range(8):
        for i in range(len(fft_res[0])):
            maxFreqMatrix[j,i] = [fft_res[j,i][8], fft_res[j,i][10], fft_res[j,i][12], fft_res[j,i][15]]
    return maxFreqMatrix

def mainDataPreparation(data):
    windowData = buildWindow(data, samplingRate, 1)
    fftData = np.zeros((8, len(windowData[0])), dtype=object)
    for j in range(8):
        for i in range(len(windowData[0])):
            fftData[j,i] = fftTransform(windowData[j,i])
    maxOut = getMaxOutFrequency(fftData)
    return maxOut

def buildOnlineLabel(maxOut):
    maxFreqByChannel = np.zeros((8, len(maxOut[0])), dtype=object)
    for j in range(8):
        for i in range(len(maxOut[0])):
            index = np.argmax(maxOut[j,i])
            maxFreqByChannel[j,i] = index+1
    return maxFreqByChannel

def buildAnswerMatrix(maxFreqByChannel, demandedFrequency):
    answerMatrix = np.zeros((8, len(maxFreqByChannel[0])), dtype=object)
    for i in range(len(maxFreqByChannel[0])):
        for j in range(8):
            answerMatrix[i,j] = demandedFrequency
    return answerMatrix

