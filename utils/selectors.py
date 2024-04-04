import numpy as np
from utils import tools


def frequencySelector(data ,freq: int):
    res = (freq - 8)/0.2
    return data[:, :, int(res), :]

def blockSelector(data, block: int):
    return data[:, :, block]

def channelSelector(data, channel: int): #Selecionar apenas os canais O1, OZ e O2 ( Linhas 61, 62 e 63  da matriz de atributos, respectivamente)
    return data[channel, :]


def parametersSelector(data, freq: int, block: int, channel: int):
    freq_data = frequencySelector(data, freq)
    block_data = blockSelector(freq_data, block)
    channel_data = channelSelector(block_data, channel)
    windowed_data = channel_data[125:-125]
    return tools.CarFilter(windowed_data)

def buildWindowedData(data, freq: int):
    channels = 2
    matrix = np.empty((5, 3), dtype=object)
    for i in range(channels):
        temp = parametersSelector(data, freq, 0, (i+60))
        for j in range(4):
            # Pegar o valor máximo das frequências de 8, 10, 12 e 15 Hz antes de colocar na matriz
            new_data = temp[j*250:(j+1)*250]
            max_data = [new_data[(8)], new_data[(10)], new_data[(12)], new_data[(15)]]
            matrix[j, i] = abs(tools.fftTransform(max_data)) #tem que pegar o abs() da fft
    return matrix



def buildFeatureMatrix(data):
    first_freq = buildWindowedData(data, 8)
    second_freq = buildWindowedData(data, 10)
    third_freq = buildWindowedData(data, 12)
    fourth_freq = buildWindowedData(data, 15)

    stacked_data = np.vstack((first_freq, second_freq, third_freq, fourth_freq))
    
    return stacked_data


# diminuir a quantidade de frequências inicialmente