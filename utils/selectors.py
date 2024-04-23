import numpy as np
from utils import tools


def frequencySelector(data ,freq: int):
    res = (freq - 8)/0.2
    return data[:, :, int(res), :]

def channelSelector(data, channel: int):
    return data[channel, :]


def parametersSelector(data, freq: int, channel: int):
    freq_data = frequencySelector(data, freq)
    channel_data = channelSelector(freq_data, channel)
    windowed_data = channel_data[125:-125]
    filtered_data = tools.CarFilter(windowed_data).reshape(-1)
    return filtered_data

