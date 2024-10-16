import numpy as np
from utils import tools, helper as hp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def frequencySelector(data ,freq: int):
    return data[:, :, freq]

def trialSelector(data, trial: int):
    return data[:, :, trial]

def channelSelector(data, channels):
    return data[0:channels, :]

def NewCarData(data, freq, channels):
    freq_data = frequencySelector(data, freq)
   
    filtered_data = np.zeros(freq_data.shape)

    for i in range(len(freq_data[1])):
        filtered_data[:, i] = tools.CarFilter(freq_data, i)
    
    channel_data = channelSelector(filtered_data, channels)
    return channel_data

def NewfftWindowAlternative(data,trials, channels, evokedFreqs, samplingRate, trainingTime):
    fft_res = np.zeros((channels*len(evokedFreqs), samplingRate*trainingTime*trials), dtype=object)
    for i in range(trainingTime*trials):
        for j in range(channels*len(evokedFreqs)):
            fft_res[j, i*samplingRate:(i+1)*samplingRate] = abs(tools.fftTransform(data[j, i*samplingRate:(i+1)*samplingRate]))
    return fft_res

# Aqui, a quantidade de for's dentro do for maior é igual a quantidade de frequências evokadas. Dá pra reduzir para um for só, mas meu cérebro não tá funcionando direito.
def NewCanalXfreqEvocada(data, trainingTime, trials, channels, evokedFreqs, samplingRate):
    temp = np.empty((trainingTime*trials*len(evokedFreqs),channels*len(evokedFreqs)), dtype=object)
    for i in range(trainingTime*trials):
        for j in range(channels):
            max_data = np.array([data[j, i*samplingRate+8], data[j, i*samplingRate+10], data[j, i*samplingRate+12], data[j, i*samplingRate+15]])
            temp[i, len(evokedFreqs)*j:len(evokedFreqs)*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+3, i*samplingRate+8], data[j+3, i*samplingRate+10], data[j+3, i*samplingRate+12], data[j+3, i*samplingRate+15]])
            temp[i+trainingTime*trials, len(evokedFreqs)*j:len(evokedFreqs)*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+6, i*samplingRate+8], data[j+6, i*samplingRate+10], data[j+6, i*samplingRate+12], data[j+6, i*samplingRate+15]])
            temp[i+2*trainingTime*trials, len(evokedFreqs)*j:len(evokedFreqs)*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+9, i*samplingRate+8], data[j+9, i*samplingRate+10], data[j+9, i*samplingRate+12], data[j+9, i*samplingRate+15]])
            temp[i+3*trainingTime*trials, len(evokedFreqs)*j:len(evokedFreqs)*(j+1)] = max_data
    return temp


def buildOnlineFeatureMatrix(data, channels, evokedFreqs, samplingRate, trainingTime, trials):

    featureMatrix = np.ones((channels*len(evokedFreqs),samplingRate*trainingTime*trials), dtype=object)

    for freq in evokedFreqs:
        evokedFreqMatrix = NewCarData(data,evokedFreqs.index(freq),channels)
        featureMatrix[evokedFreqs.index(freq)*channels:(evokedFreqs.index(freq)+1)*channels, :] = evokedFreqMatrix

    windowedData = NewfftWindowAlternative(featureMatrix, trials, channels, evokedFreqs, samplingRate, trainingTime)
    
    # Escolhemos 1 canal para cada frequência evocada e plotamos o sinal.
    x = np.arange(30)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(x, windowedData[0, samplingRate:samplingRate+30], label=f'{evokedFreqs[0]}Hz')
    axs[0, 0].set_title(f'{evokedFreqs[0]}Hz')
    axs[0, 0].scatter(evokedFreqs, [windowedData[0, samplingRate+8], windowedData[0, samplingRate+10], windowedData[0, samplingRate+12], windowedData[0, samplingRate+15]], color='red')
    
    axs[0, 1].plot(x, windowedData[3, samplingRate:samplingRate+30], label=f'{evokedFreqs[1]}Hz')
    axs[0, 1].set_title(f'{evokedFreqs[1]}Hz')
    axs[0, 1].scatter(evokedFreqs, [windowedData[3, samplingRate+8], windowedData[3, samplingRate+10], windowedData[3, samplingRate+12], windowedData[3, samplingRate+15]], color='red')
    
    axs[1, 0].plot(x, windowedData[7, samplingRate:samplingRate+30], label=f'{evokedFreqs[2]}Hz')
    axs[1, 0].set_title(f'{evokedFreqs[2]}Hz')
    axs[1, 0].scatter(evokedFreqs, [windowedData[7, samplingRate+8], windowedData[7, samplingRate+10], windowedData[7, samplingRate+12], windowedData[7, samplingRate+15]], color='red')
    
    axs[1, 1].plot(x, windowedData[10, samplingRate:samplingRate+30], label=f'{evokedFreqs[3]}Hz')
    axs[1, 1].set_title(f'{evokedFreqs[3]}Hz')
    axs[1, 1].scatter(evokedFreqs, [windowedData[10, samplingRate+8], windowedData[10, samplingRate+10], windowedData[10, samplingRate+12], windowedData[10, samplingRate+15]], color='red')
    for ax in axs.flat:
        ax.set(xlabel='Frequência (Hz)', ylabel='Amplitude do sinal')
        ax.legend()
    plt.tight_layout()
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude do sinal')
    plt.show()

    attributeMatrix = NewCanalXfreqEvocada(windowedData, trainingTime, trials, channels, evokedFreqs, samplingRate)

    ones = np.ones((trainingTime*trials*len(evokedFreqs),1), dtype=object)
    dataWithOnes = np.hstack((attributeMatrix,ones))
    
    print(f"Matriz de atributos: {dataWithOnes.shape}")
    return dataWithOnes