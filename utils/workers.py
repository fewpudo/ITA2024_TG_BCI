limport numpy as np
from utils import selectors, tools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Criar uma função que mostra qual o maior indíce da matriz de estimated_y de probabilidades
# Melhorar essa função
def buildValidationAndTestMatrix(data, y):
    testMatrix, validationMatrix, yTest, yValidation = train_test_split(data, y, test_size=0.2, stratify=y)
    return testMatrix, validationMatrix, yTest, yValidation


def buildFeatureMatrix(data):
    featureMatrix = np.ones((12,7500), dtype=object)
    EightHzMatrix = selectors.CarData(data,0)
    TenHzMatrix = selectors.CarData(data,2)
    TwelveHzMatrix = selectors.CarData(data,4)
    FifteenHzMatrix = selectors.CarData(data,7)
    featureMatrix[0:3, :] = EightHzMatrix
    featureMatrix[3:6, :] = TenHzMatrix
    featureMatrix[6:9, :] = TwelveHzMatrix
    featureMatrix[9:12, :] = FifteenHzMatrix
    fftTransformMatrix = tools.fftTransform(featureMatrix)
    print(fftTransformMatrix[11,0])
    return fftTransformMatrix

def buildLabelMatrix(data):
    y = np.ones((13,7500), dtype=object)*-1
    for i in range(len(data[1])):
        for j in range(len(data[0])):


    return