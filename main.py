from utils import selectors
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


data = scipy.io.loadmat('subjects/S1.mat')
featureMatrix = workers.buildFeatureMatrix(data['data'])



# Primeira parte do TG é explicar o sistema BCI-SSVEP, apresentar os algoritmos e colocar os dados da simulação.
# Segunda parte é fazer o sistema Online -> Usar a toca e ler online.