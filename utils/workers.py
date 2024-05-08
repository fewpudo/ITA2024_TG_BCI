import numpy as np
from utils import selectors, tools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Criar uma função que mostra qual o maior indíce da matriz de estimated_y de probabilidades
# Melhorar essa função
# Matrizes tem dimensão (1250 amostras) x (3 canais * 6 trials * 4 frequências + 1 coluna de uns) -> Posso ajustar rápido com a professora!
def buildValidationAndTestMatrix(data):
    y = buildLabelMatrix(data)
    testMatrix, validationMatrix, yTest, yValidation = train_test_split(data.T, y.T,random_state= 100, test_size=0.2) 
    return testMatrix, validationMatrix, yTest, yValidation


def buildFeatureMatrix(data):
    featureMatrix = np.ones((18*4,1250), dtype=object)
    EightHzMatrix = selectors.CarData(data,0)
    TenHzMatrix = selectors.CarData(data,2)
    TwelveHzMatrix = selectors.CarData(data,4)
    FifteenHzMatrix = selectors.CarData(data,7)
    featureMatrix[0:18, :] = EightHzMatrix
    featureMatrix[18:36, :] = TenHzMatrix
    featureMatrix[36:54, :] = TwelveHzMatrix
    featureMatrix[54:72, :] = FifteenHzMatrix
    ones = np.ones((1,1250), dtype=object)
    dataWithOnes = np.vstack((featureMatrix,ones))
    fftTransformMatrix = abs(tools.fftTransform(dataWithOnes))
    # Após a FFT o gráfico tá estranho
    # plt.plot(fftTransformMatrix[5,:])
    # plt.show()
    return fftTransformMatrix

def buildLabelMatrix(data):
    y = np.ones((12*6+1,1250), dtype=object)*-1
    for i in range(len(data[1])):
        index = np.argmax(data[:,i])
        y[index,i] = 1
    return y

def buildWMatrix(testMatrix, yTest):
    pinvX = np.linalg.pinv(testMatrix.astype(np.float64))
    W = np.matmul(pinvX, yTest)
    print(W.shape)
    return W

def Acuraccy(validationMatrix, W, yValidation):
    estimated_y = np.matmul(W, validationMatrix.T)
    estimated_y = np.sign(estimated_y)
    print(estimated_y[:,0])
    print(yValidation[0,:])
    print(estimated_y.shape)
    print(yValidation.shape)
    acc = np.sum(estimated_y == yValidation.T)/(250*73)
    print(acc)

