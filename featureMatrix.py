import numpy as np
from utils import tools, helper as hp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def frequencySelector(data ,freq: int):
    return data[:, :, freq]

def trialSelector(data, trial: int):
    return data[:, :, trial]

def channelSelector(data, channels):
    return data[60:60+channels, :]

def NewCarData(data, freq, channels, samplingRate, trainingTime):
    freq_data = frequencySelector(data, freq)
   
    filtered_data = np.zeros(freq_data.shape)

    for i in range(len(freq_data[1])):
        filtered_data[:, i] = tools.CarFilter(freq_data, i)
    
    channel_data = channelSelector(filtered_data, channels)
    return channel_data

def NewfftWindowAlternative(data,trials, channels, evokedFreqs, samplingRate, trainingTime):
    fft_res = np.zeros((channels*evokedFreqs, samplingRate*trainingTime*trials), dtype=object)
    for i in range(trainingTime*trials):
        for j in range(channels*evokedFreqs):
            fft_res[j, i*samplingRate:(i+1)*samplingRate] = abs(tools.fftTransform(data[j, i*samplingRate:(i+1)*samplingRate]))
    return fft_res

# Aqui, a quantidade de for's dentro do for maior é igual a quantidade de frequências evokadas. Dá pra reduzir para um for só, mas meu cérebro não tá funcionando direito.
def NewCanalXfreqEvocada(data, trainingTime, trials, channels, evokedFreqs, samplingRate):
    temp = np.empty((trainingTime*trials*evokedFreqs,channels*evokedFreqs), dtype=object)
    for i in range(trainingTime*trials):
        for j in range(channels):
            max_data = np.array([data[j, i*samplingRate+8], data[j, i*samplingRate+10], data[j, i*samplingRate+12], data[j, i*samplingRate+15]])
            temp[i, evokedFreqs*j:evokedFreqs*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+3, i*samplingRate+8], data[j+3, i*samplingRate+10], data[j+3, i*samplingRate+12], data[j+3, i*samplingRate+15]])
            temp[i+trainingTime*trials, evokedFreqs*j:evokedFreqs*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+6, i*samplingRate+8], data[j+6, i*samplingRate+10], data[j+6, i*samplingRate+12], data[j+6, i*samplingRate+15]])
            temp[i+2*trainingTime*trials, evokedFreqs*j:evokedFreqs*(j+1)] = max_data
        for j in range(channels):
            max_data = np.array([data[j+9, i*samplingRate+8], data[j+9, i*samplingRate+10], data[j+9, i*samplingRate+12], data[j+9, i*samplingRate+15]])
            temp[i+3*trainingTime*trials, evokedFreqs*j:evokedFreqs*(j+1)] = max_data
    return temp


# Função mais parametrizada para criar a matriz de atributos.
def buildOnlineFeatureMatrix(data, channels, evokedFreqs, samplingRate, trainingTime, trials):

    featureMatrix = np.ones((channels*evokedFreqs,samplingRate*trainingTime*trials), dtype=object)

    EightHzMatrix = NewCarData(data,0,channels,samplingRate,trainingTime)
    TenHzMatrix = NewCarData(data,2,channels,samplingRate,trainingTime)
    TwelveHzMatrix = NewCarData(data,4,channels,samplingRate,trainingTime)
    FifteenHzMatrix = NewCarData(data,7,channels,samplingRate,trainingTime)

    # Particular para a quantidade de frequências evocadas.
    featureMatrix[0:channels, :] = EightHzMatrix
    featureMatrix[channels:2*channels, :] = TenHzMatrix
    featureMatrix[2*channels:3*channels, :] = TwelveHzMatrix
    featureMatrix[3*channels:4*channels, :] = FifteenHzMatrix

    windowedData = NewfftWindowAlternative(featureMatrix, trials, channels, evokedFreqs, samplingRate, trainingTime)
    # Plotting the graph
    x = np.arange(30)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(x, windowedData[0, samplingRate:samplingRate+30], label='8Hz')
    axs[0, 0].set_title('8Hz')
    axs[0, 0].scatter([8, 10, 12, 15], [windowedData[0, samplingRate+8], windowedData[0, samplingRate+10], windowedData[0, samplingRate+12], windowedData[0, samplingRate+15]], color='red')
    
    axs[0, 1].plot(x, windowedData[3, samplingRate:samplingRate+30], label='10Hz')
    axs[0, 1].set_title('10Hz')
    axs[0, 1].scatter([8, 10, 12, 15], [windowedData[3, samplingRate+8], windowedData[3, samplingRate+10], windowedData[3, samplingRate+12], windowedData[3, samplingRate+15]], color='red')
    
    axs[1, 0].plot(x, windowedData[7, samplingRate:samplingRate+30], label='12Hz')
    axs[1, 0].set_title('12Hz')
    axs[1, 0].scatter([8, 10, 12, 15], [windowedData[7, samplingRate+8], windowedData[7, samplingRate+10], windowedData[7, samplingRate+12], windowedData[7, samplingRate+15]], color='red')
    
    axs[1, 1].plot(x, windowedData[10, samplingRate:samplingRate+30], label='15Hz')
    axs[1, 1].set_title('15Hz')
    axs[1, 1].scatter([8, 10, 12, 15], [windowedData[10, samplingRate+8], windowedData[10, samplingRate+10], windowedData[10, samplingRate+12], windowedData[10, samplingRate+15]], color='red')
    for ax in axs.flat:
        ax.set(xlabel='Frequência (Hz)', ylabel='Amplitude do sinal')
        ax.legend()
    plt.tight_layout()
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude do sinal')
    plt.title('FFTs')
    plt.show()

    attributeMatrix = NewCanalXfreqEvocada(windowedData, trainingTime, trials, channels, evokedFreqs, samplingRate)

    ones = np.ones((trainingTime*trials*evokedFreqs,1), dtype=object)
    dataWithOnes = np.hstack((attributeMatrix,ones))
    
    print(f"Matriz de atributos: {dataWithOnes.shape}")
    return dataWithOnes