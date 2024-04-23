import numpy as np
from utils import tools


def frequencySelector(data ,freq: int):
    res = (freq - 8)/0.2
    return data[:, :, int(res), :]

def channelSelector(data, channel: int):
    return data[channel, :]


def parametersSelector(data, freq: int, channel: int):
    freq_data = frequencySelector(data, freq)
    channel_data = channelSelector(freq_data, channel)
    windowed_data = channel_data[125:-125]
    filtered_data = tools.CarFilter(windowed_data).reshape(-1)
    return filtered_data

def buildWindowedData(data, freq: int):
    channels = 3
    windows = 30
    matrix = np.empty((30, 12), dtype=object)
    full_data_matrix = np.empty((30, 3), dtype=object)

    for i in range(channels):
        temp = parametersSelector(data, freq, (i+60))

        for j in range(windows):
            new_data = abs(tools.fftTransform(temp[j*250:(j+1)*250]))
            max_data = [new_data[(8)], new_data[(10)], new_data[(12)], new_data[(15)]]
            full_data_matrix[j,i] = new_data
            matrix[j, i] = max_data
            
    return matrix, full_data_matrix



def buildFeatureMatrix(data):
    first_freq, full_first_freq = buildWindowedData(data, 8)
    second_freq, full_second_freq = buildWindowedData(data, 10)
    third_freq, full_third_freq = buildWindowedData(data, 12)
    fourth_freq, full_fourth_freq = buildWindowedData(data, 15)

    stacked_data = np.vstack((first_freq, second_freq, third_freq, fourth_freq))
    full_stacked_data = np.vstack((full_first_freq, full_second_freq, full_third_freq, full_fourth_freq))
    
    return stacked_data, full_stacked_data
