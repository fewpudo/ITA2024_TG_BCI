
def frequencySelector(data ,freq: int):
    res = (freq - 8)/0.2
    return data[:, :, int(res), :]

def blockSelector(data, block: int):
    return data[:, :, block]

def channelSelector(data, channel: int):
    return data[channel, :]


def parametersSelector(data, freq: int, block: int):
    freq_data = frequencySelector(data, freq)
    block_data = blockSelector(freq_data, block)
    return block_data

def buildFeatureMatrix(data, freq: int, block: int):
    return parametersSelector(data, freq, block).T