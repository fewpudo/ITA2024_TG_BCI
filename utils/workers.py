import numpy as np
from utils import selectors, tools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Criar uma função que mostra qual o maior indíce da matriz de estimated_y de probabilidades
# Melhorar essa função
def buildValidationAndTestMatrix(data, y):
    testMatrix, validationMatrix, yTest, yValidation = train_test_split(data, y, test_size=0.2, stratify=y)
    return testMatrix, validationMatrix, yTest, yValidation

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


def buildWMatrix(data, y_matrix):
    pinvx = np.linalg.pinv(data.astype(np.float64))
    w_matrix = np.matmul(pinvx, y_matrix)
    return w_matrix



 # TODO: turn this into a freq depender
def classifier(data):
    y8, y10, y12, y15 = buildYMatrix()
    test8Matrix, validation8Matrix, y8Test, y8Validation = buildValidationAndTestMatrix(data,y8)
    test10Matrix, validation10Matrix, y10Test, y10Validation = buildValidationAndTestMatrix(data,y10)
    test12Matrix, validation12Matrix, y12Test, y12Validation = buildValidationAndTestMatrix(data,y12)
    test15Matrix, validation15Matrix, y15Test, y15Validation = buildValidationAndTestMatrix(data,y15)
    w8, w10, w12, w15 = buildWMatrix(test8Matrix, y8Test), buildWMatrix(test10Matrix, y10Test), buildWMatrix(test12Matrix, y12Test), buildWMatrix(test15Matrix, y15Test)

    estimated_y8 = np.matmul(validation8Matrix, w8)
    estimated_y10 = np.matmul(validation10Matrix, w10)
    estimated_y12 = np.matmul(validation12Matrix, w12)
    estimated_y15 = np.matmul(validation15Matrix, w15)
    


    estimated_y8[estimated_y8 < 0] = -1
    estimated_y8[estimated_y8 > 0] = 1
    estimated_y10[estimated_y10 < 0] = -1
    estimated_y10[estimated_y10 > 0] = 1
    estimated_y12[estimated_y12 < 0] = -1
    estimated_y12[estimated_y12 > 0] = 1
    estimated_y15[estimated_y15 < 0] = -1
    estimated_y15[estimated_y15 > 0] = 1
    acuracia8 = np.sum(estimated_y8 == y8Validation) / len(y8Validation)
    acuracia10 = np.sum(estimated_y10 == y10Validation) / len(y10Validation)
    acuracia12 = np.sum(estimated_y12 == y12Validation) / len(y12Validation)
    acuracia15 = np.sum(estimated_y15 == y15Validation) / len(y15Validation)
    return acuracia8, acuracia10, acuracia12, acuracia15
    


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
    first_freq, full_first_freq = buildWindowedData(data, 0)
    second_freq, full_second_freq = buildWindowedData(data, 2)
    third_freq, full_third_freq = buildWindowedData(data, 4)
    fourth_freq, full_fourth_freq = buildWindowedData(data, 7)

    stacked_data = np.vstack((first_freq, second_freq, third_freq, fourth_freq))
    full_stacked_data = np.vstack((full_first_freq, full_second_freq, full_third_freq, full_fourth_freq))
    
    return stacked_data, full_stacked_data


