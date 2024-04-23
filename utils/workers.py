import numpy as np
from utils import selectors, tools

# verify if the data is correct
# This function gets the data randomly and separates it into validation and test data 
def buildValidationAndTestMatrix(data):
    validationMatrix = data[np.random.choice(data.shape[0], 24, replace=False), :]
    testMatrix = data[np.random.choice(np.setdiff1d(np.arange(data.shape[0]), validationMatrix[:, 0]), 96, replace=False), :]
    return validationMatrix, testMatrix

# TODO: fix this logic
def buildY8Matrix():
    y_matrix = np.empty((120, 1))
    y_matrix[:30] = 1
    y_matrix[30:60] = -1
    y_matrix[60:90] = -1
    y_matrix[90:] = -1
    return y_matrix

def buildY10Matrix():
    y_matrix = np.empty((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = 1
    y_matrix[60:90] = -1
    y_matrix[90:] = -1
    return y_matrix

def buildY12Matrix():
    y_matrix = np.empty((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = -1
    y_matrix[60:90] = 1
    y_matrix[90:] = -1
    return y_matrix

def buildY15Matrix():
    y_matrix = np.empty((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = -1
    y_matrix[60:90] = -1
    y_matrix[90:] = 1
    return y_matrix

def buildYMatrix():
    y8 = buildY8Matrix()
    y10 = buildY10Matrix()
    y12 = buildY12Matrix()
    y15 = buildY15Matrix()
    return y8, y10, y12, y15


def buildW8Matrix(data):
    w_matrix = np.empty((13, 1))
    pinvx = np.linalg.pinv(data)
    y_matrix = buildY8Matrix()
    w_matrix = np.dot(pinvx, y_matrix)
    return w_matrix







# É possível fazer tudo dentro do for da buildWindowedData, mas acho que é melhor ficar separado para que eu consiga construir a matriz de formas diferentes
# Adiciona 1's na coluna final
def refactorMatrix(data):
    window = 120
    channels = 3
    refactored_matrix = np.empty((120, 13), dtype=object)
    for i in range(window):
        for j in range(channels):
            refactored_matrix[i, j*4:(j+1)*4] = data[i, j]
            refactored_matrix[i, 12] = 1
    return refactored_matrix

def buildWindowedData(data, freq: int):
    channels = 3
    windows = 30
    matrix = np.empty((30, 3), dtype=object)
    full_data_matrix = np.empty((30, 3), dtype=object)

    for i in range(channels):
        temp = selectors.parametersSelector(data, freq, (i+60))

        for j in range(windows):
            new_data = abs(tools.fftTransform(temp[j*250:(j+1)*250]))
            full_data_matrix[j,i] = new_data
            max_data = [new_data[(8)], new_data[(10)], new_data[(12)], new_data[(15)]]
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


