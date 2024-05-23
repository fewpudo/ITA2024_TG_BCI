from utils import selectors
import scipy
import matplotlib.pyplot as plt
from utils import workers
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd



# dados
# [64, 1500, 40, 6]
# 64 canais, 1500 amostras, 40 frequências, 6 testes diferentes
# Particularidades desse dataset: Já vem filtrado com notch e o passa-baixas e, o formato dos dados exibidos é [6, 40, 1500, 64]
# No aparelho será preciso aplicar o filtro notch também, não vem automaticamente. 

# Pegar para o documento do TG, índice [2,1] já funciona.
# Plotar imagens filtrados com CAR e filtrado sem CAR.
# Verificar se o filtro notch é implementado pela placa e também se é disponibilizado pela API.

# Indivíduo 15 é muito bom para pegar os dados, ele tem uma acurácia muito perto 100% no teste.
acc = np.zeros((35,4), dtype=object)
for k in range(35):
    subject = f"subjects/S{k+1}.mat"
    data = scipy.io.loadmat(subject)
    featureMatrix = workers.buildFeatureMatrix(data['data'])
    testMatrix, validationMatrix, yTest, yValidation = workers.buildValidationAndTestMatrix(featureMatrix)
    WMatrix = workers.buildWMatrix(testMatrix, yTest)
    acc[k,:] = workers.AcuraccyByFreq(validationMatrix, WMatrix, yValidation)

# Construir a tabela com 4 colunas e 35 linhas
# Exportar a tabela para um arquivo xls

table = acc
column_titles = ['8hz', '10Hz', '12Hz', '15Hz']

# Imprimir a tabela
for i in range(len(table)):
    for j in range(len(table[i])):
        table[i][j] = "{:.2f}".format(float(table[i][j]))
        
df = pd.DataFrame(table, columns=column_titles)
df.to_csv('C:/Users/T-GAMER/Desktop/ITA/TGBCI/table.csv', index=True)
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

# Primeira parte do TG é explicar o sistema BCI-SSVEP, apresentar os algoritmos e colocar os dados da simulação.
# Segunda parte é fazer o sistema Online -> Usar a toca e ler online.