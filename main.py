import scipy
import matplotlib.pyplot as plt
from utils import selectors, workers



# dados
# [64, 1500, 40, 6]
# 64 canais, 1500 amostras, 40 frequências, 6 testes diferentes
# Particularidades desse dataset: Já vem filtrado com notch e o passa-baixas e, o formato dos dados exibidos é [6, 40, 1500, 64]
# No aparelho será preciso aplicar o filtro notch também, não vem automaticamente. 

# Pegar para o documento do TG, índice [2,1] já funciona.
# Plotar imagens filtrados com CAR e filtrado sem CAR.
# Verificar se o filtro notch é implementado pela placa e também se é disponibilizado pela API.


# func trainTestSplit(X, y, test_size=0.2, random_state=None) -> É bom já ter construído o Y antes de chamar essa função.

# W tem 13x1 e Y tem a mesmo número de linhas de X

data = scipy.io.loadmat('subjects/S1.mat')
featureMatrix, full_featureMatrix = selectors.buildFeatureMatrix(data['data'])
fixedMatrix = selectors.refactorMatrix(featureMatrix)
validationMatrix, testMatrix = workers.buildValidationAndTestMatrix(fixedMatrix)
print(validationMatrix.shape)
print(testMatrix.shape)



# print(fixedMatrix.shape)
# print(fixedMatrix[2, 12])
# print(featureMatrix.shape)
# print(featureMatrix[2, 1])

# Próximo passo: Implementar o modelo de classificação






