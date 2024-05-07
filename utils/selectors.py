import numpy as np
from utils import tools

# 8 -> 0, 10 -> 2, 12 -> 4, 15 -> 7
def frequencySelector(data ,freq: int):
    return data[:, :, freq, :]

def channelSelector(data, channel: int):
    return data[channel, :]

# O problema é que eu estou usando o filtro CAR por cada canal separadamente. Preciso usar todos os canais juntos. Fazer separado por frequência e trial.
# função CAR vai receber uma matriz de 1250x64 de cada frequência e cada trial
# Primeiro passo eu seleciono a frequência, mata as 250 linhas 125 inicias e 125 finais, depois eu seleciono a trial e depois eu aplico o filtro CAR.
# Filtro CAR é pra ser implementado em cada amostra, ou seja, em cada linha da minha matriz. Subtrair a média da linha da matriz.
def parametersSelector(data, freq: int, channel: int):
    freq_data = frequencySelector(data, freq)
    channel_data = channelSelector(freq_data, channel)
    print(channel_data.shape)
    windowed_data = channel_data[125:-125]
    filtered_data = tools.CarFilter(windowed_data).reshape(-1)
    return filtered_data

