import scipy
import matplotlib.pyplot as plt
from utils import workers
import numpy as np
from sklearn.svm import LinearSVC



# dados
# [64, 1500, 40, 6]
# 64 canais, 1500 amostras, 40 frequências, 6 testes diferentes
# Particularidades desse dataset: Já vem filtrado com notch e o passa-baixas e, o formato dos dados exibidos é [6, 40, 1500, 64]
# No aparelho será preciso aplicar o filtro notch também, não vem automaticamente. 

# Pegar para o documento do TG, índice [2,1] já funciona.
# Plotar imagens filtrados com CAR e filtrado sem CAR.
# Verificar se o filtro notch é implementado pela placa e também se é disponibilizado pela API.


# func trainTestSplit(X, y, test_size=0.2, random_state=None) -> É bom já ter construído o Y antes de chamar essa função. Usar o stratify on aqui! pois preserva a proporção dos dados.
# Selecionar 80% de cada frequência para treino e 20% para teste, de forma aleatória. Pois está errado. -> done
# Separar a matriz de label também. -> done
# Organizar as matrizes de treino e validação de forma que fique mais simples criar os Y's. -> done
# Criar a matriz de confusão = Quantas vezes o classificador acertou e quantas vezes errou.
# Provavelmente tem a alguma função pronta que faz isso.
# Calcular a acurácia


data = scipy.io.loadmat('subjects/S1.mat')
featureMatrix, full_featureMatrix = workers.buildFeatureMatrix(data['data'])
fixedMatrix = workers.refactorMatrix(featureMatrix)
acuracy8, acuracy10, acuracy12, acuracy15 = workers.classifier(fixedMatrix)
# plt.plot(fixedMatrix[:, 1])
# plt.show()
# print(acuracy8)
# print(acuracy10)
# print(acuracy12)
# print(acuracy15)



# print(fixedMatrix.shape)
# print(fixedMatrix[2, 12])
# print(featureMatrix.shape)
# print(featureMatrix[2, 1])






# Primeira parte do TG é explicar o sistema BCI-SSVEP, apresentar os algoritmos e colocar os dados da simulação.
# Segunda parte é fazer o sistema Online -> Usar a toca e ler online.