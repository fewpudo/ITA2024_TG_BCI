import csv
import numpy as np
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
especific_data = selectors.frequencySelector(data['data'], 10)
filtered_data = tools.CarFilter(especific_data)




