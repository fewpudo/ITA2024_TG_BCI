import scipy
import matplotlib.pyplot as plt
from utils import workers
import numpy as np
from sklearn.svm import LinearSVC
import featureMatrix as ft

# mock 
channels = 3
evokedFreqs = 4
samplingRate = 250
trials = 6
trainningTime = 5

def buildClassifier():
    acc = np.zeros((35,4), dtype=object)
    for k in range(1):
        subject = f"subjects/S{1}.mat"
        data = scipy.io.loadmat(subject)
        labelMatrix = workers.buildLabelMatrix(trainningTime, trials, evokedFreqs)
        featureMatrix = ft.buildOnlineFeatureMatrix(data['data'], channels, evokedFreqs,samplingRate, trainningTime, trials)
        testMatrix, validationMatrix, yTest, yValidation = workers.buildValidationAndTestMatrix(featureMatrix, labelMatrix)

        WMatrix = workers.buildWMatrix(testMatrix, yTest)
        acc[k,:] = workers.AcuraccyByFreq(validationMatrix, WMatrix, yValidation)

    table = acc
    column_titles = ['8hz', '10Hz', '12Hz', '15Hz']

    # Imprimir a tabela
    for i in range(len(table)):
        for j in range(len(table[i])):
            table[i][j] = "{:.2f}".format(float(table[i][j]))
            

    # Color cells with values above 75 green and round to two decimal places
    for i in range(0, len(table)):
        for j in range(0, len(table[i])):
            if float(table[i][j]) >= 75:
                table[i][j] = '\033[92m' + "{:.2f}".format(float(table[i][j])) + '\033[0m'
            elif 45 <= float(table[i][j]) <= 74:
                table[i][j] = '\033[93m' + "{:.2f}".format(float(table[i][j])) + '\033[0m'
            else:
                table[i][j] = '\033[91m' + "{:.2f}".format(float(table[i][j])) + '\033[0m'

    # Imprimir a tabela formatada
    for row in table:
        print('\t'.join(str(cell) for cell in row))

def buildWForOnline(data, channels, evokedFreqs, samplingRate, trainningTime, trials):
    labelMatrix = workers.buildLabelMatrix(trainningTime, trials, evokedFreqs)
    featureMatrix = ft.buildOnlineFeatureMatrix(data, channels, evokedFreqs,samplingRate, trainningTime, trials)
    testMatrix, validationMatrix, yTest, yValidation = workers.buildValidationAndTestMatrix(featureMatrix, labelMatrix)
    WMatrix = workers.buildWMatrix(testMatrix, yTest)
    
    return WMatrix

# Função para utilizar o classificador criado no treinamento
def classify(classifier, x_entrada):
    y_pred = np.matmul(x_entrada, classifier)
    for i in range(y_pred.shape[0]):
        index = np.argmax(y_pred[i,:])
        y_pred[i,index] = 1
        y_pred[i,y_pred[i,:] != 1] = -1
    return y_pred