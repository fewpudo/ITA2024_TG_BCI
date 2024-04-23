import scipy
import matplotlib.pyplot as plt
from utils import selectors, tools



# dados
# [64, 1500, 40, 6]
# 64 canais, 1500 amostras, 40 frequências, 6 testes diferentes
# Particularidades desse dataset: Já vem filtrado com notch e o passa-baixas e, o formato dos dados exibidos é [6, 40, 1500, 64]
# No aparelho será preciso aplicar o filtro notch também, não vem automaticamente. 

# Pegar para o documento do TG, índice [2,1] já funciona.
# Plotar imagens filtrados com CAR e filtrado sem CAR.
# Verificar se o filtro notch é implementado pela placa e também se é disponibilizado pela API.

# Separar cada vetor para cada canal, O1(8), O1(10), O1(12), O1(15) , OZ(8), OZ(10)...
# y = ax + 1x = (a+1)x
# matrix A tem 13 colunas ao inves de 12, sendo a última coluna uma coluna de 1s (13 x 120)
# separar minha matriz de atributos e colocar uma coluna de 1s no final
# func trainTestSplit(X, y, test_size=0.2, random_state=None) -> É bom já ter construído o Y antes de chamar essa função.
# Gerar um handle para pegar a quantidade de amostra, não posso treinar com tudo pois tenho que validar. Escolher de forma aleatória.
# 6 amostras de cada frequÊncia para validar e 24 amostras para treinar.
# Basicamente separar minha matriz X em X_treino e X_validação, sendo 96x13 e 24x13 respectivamente.
# W tem 13x1 e Y tem a mesmo número de linhas de X

data = scipy.io.loadmat('subjects/S1.mat')
featureMatrix, full_featureMatrix = selectors.buildFeatureMatrix(data['data'])
print(full_featureMatrix.shape)
print(featureMatrix.shape)
print(featureMatrix[2, 1])


# Próximo passo: Implementar o modelo de classificação






