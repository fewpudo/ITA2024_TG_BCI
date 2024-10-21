import numpy as np
from . import helper
from utils import tools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def buildValidationAndTestMatrix(data, labelMatrix):
    testMatrix, validationMatrix, yTest, yValidation = train_test_split(data, labelMatrix,random_state=30, test_size=0.2) 
    print(f"Matriz de teste: {testMatrix.shape}")
    print(f"Matriz de labels de validacao: {yValidation.shape}")
    return testMatrix, validationMatrix, yTest, yValidation

def buildLabelMatrix(trainingTime, trials, evokedFreqs):
    y = np.ones((trainingTime*trials*len(evokedFreqs),len(evokedFreqs)), dtype=object)*-1
    y[0:trainingTime*trials,0] = 1
    y[trainingTime*trials:2*trainingTime*trials,1] = 1
    y[2*trainingTime*trials:3*trainingTime*trials,2] = 1
    y[3*trainingTime*trials:4*trainingTime*trials,3] = 1
    print(f"Matriz de labels: {y.shape}")
    return y

def buildWMatrix(testMatrix, yTest):
    pinvX = np.linalg.pinv(testMatrix.astype(np.float64))
    W = np.matmul(pinvX, yTest)
    print(f"Matriz W: {W.shape}")
    return W

def Acuraccy(validationMatrix, W, yValidation):
    print(f"Matriz W: {W.shape}")
    print(f"Matriz de validação: {validationMatrix.shape}")
    
    estimated_y = np.matmul(validationMatrix,W)
    for i in range(estimated_y.shape[0]):
        index = np.argmax(estimated_y[i,:])
        estimated_y[i,index] = 1
        estimated_y[i,estimated_y[i,:] != 1] = -1
    print(f"Matriz de labels estimadas: {estimated_y.shape}")
    print(f"Matriz de labels reais: {yValidation.shape}")
    accuracy = np.mean(estimated_y == yValidation)
    print(f"Accuracy: {accuracy*100}")

def AcuraccyByFreq(validationMatrix, W, yValidation):
    accuracy = np.array([0,0,0,0], dtype=object)
    estimated_y = np.matmul(validationMatrix,W)
    for i in range(estimated_y.shape[0]):
        index = np.argmax(estimated_y[i,:])
        estimated_y[i,index] = 1
        estimated_y[i,estimated_y[i,:] != 1] = -1
    for i in range(4):
        accuracy[i] = (np.sum(estimated_y[:,i] == yValidation[:,i])/len(yValidation[:,i]))*100
        
    print(f"Accuracy 8Hz: {accuracy[0]}")
    print(f"Accuracy 10Hz: {accuracy[1]}")
    print(f"Accuracy 12Hz: {accuracy[2]}")
    print(f"Accuracy 15Hz: {accuracy[3]}")
    return accuracy

def CanalXfreqEvocada(data):
    temp = np.empty((120,12), dtype=object)
    for i in range(30):
        for j in range(3):
            max_data = np.array([data[j, i*250+8], data[j, i*250+10], data[j, i*250+12], data[j, i*250+15]])
            temp[i, 4*j:4*(j+1)] = max_data
        for j in range(3):
            max_data = np.array([data[j+3, i*250+8], data[j+3, i*250+10], data[j+3, i*250+12], data[j+3, i*250+15]])
            temp[i+30, 4*j:4*(j+1)] = max_data
        for j in range(3):
            max_data = np.array([data[j+6, i*250+8], data[j+6, i*250+10], data[j+6, i*250+12], data[j+6, i*250+15]])
            temp[i+60, 4*j:4*(j+1)] = max_data
        for j in range(3):
            max_data = np.array([data[j+9, i*250+8], data[j+9, i*250+10], data[j+9, i*250+12], data[j+9, i*250+15]])
            temp[i+90, 4*j:4*(j+1)] = max_data
    return temp

def fftWindowAlternative(data):
    fft_res = np.zeros((12, 7500), dtype=object)
    for i in range(30):
        for j in range(12):
            fft_res[j, i*250:(i+1)*250] = abs(tools.fftTransform(data[j, i*250:(i+1)*250]))
    return fft_res

