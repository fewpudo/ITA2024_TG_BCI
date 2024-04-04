import scipy
import matplotlib.pyplot as plt
from utils import selectors, tools



# dados
# [64, 1500, 40, 6]
# 64 canais, 1500 amostras, 40 frequências, 6 testes diferentes
# Canal 1, Todas as amostras, frequência de 8Hz, todos os testes
# example = data['data'][0,:,0,:]
# print(example)


data = scipy.io.loadmat('subjects/S1.mat')
featureMatrix, full_featureMatrix = selectors.buildFeatureMatrix(data['data'])
print(full_featureMatrix.shape)
print(featureMatrix.shape)
plt.plot(full_featureMatrix[3, 0]) 
plt.xlim(5, 30)
plt.show()









