
def frequencySelector(data ,freq: int):
    res = (freq - 8)/2
    return data[:, :, int(res), :]

def blockSelector(data, block: int):
    return data[:, :, block]

def channelSelector(data, channel: int):
    return data[channel, :]


def parametersSelector(data, freq: int, block: int, channel: int):
    freq_data = frequencySelector(data, freq)
    block_data = blockSelector(freq_data, block)
    channel_data = channelSelector(block_data, channel)
    return channel_data