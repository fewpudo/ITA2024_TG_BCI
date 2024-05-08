import numpy as np
from utils import tools

# 8 -> 0, 10 -> 2, 12 -> 4, 15 -> 7
def frequencySelector(data ,freq: int):
    return data[:, :, freq, :]

def trialSelector(data, trial: int):
    return data[:, :, trial]

def channelSelector(data):
    return data[60:63, :]

# O problema é que eu estou usando o filtro CAR por cada canal separadamente. Preciso usar todos os canais juntos. Fazer separado por frequência e trial. -> Done
# função CAR vai receber uma matriz de 1250x64 de cada frequência e cada trial -> Done
# Primeiro passo eu seleciono a frequência, mata as 250 linhas 125 inicias e 125 finais, depois eu seleciono a trial e depois eu aplico o filtro CAR. -> Done
# Filtro CAR é pra ser implementado em cada amostra, ou seja, em cada linha da minha matriz. Subtrair a média da linha da matriz. -> Done
def CarData(data, freq: int):
    freq_data = frequencySelector(data, freq)
    clean_data = freq_data[: , 125:-125, :]
    trial_data = np.zeros((64*6, 1250)) 

    for i in range(5):
        trial_data[:, i*1250:(i+1)*1250] = trialSelector(clean_data, i)
    
    filtered_data = np.zeros(trial_data.shape)

    for i in range(len(trial_data[1])):
        filtered_data[:, i] = tools.CarFilter(trial_data, i)
    channel_data = channelSelector(filtered_data)

    return channel_data

