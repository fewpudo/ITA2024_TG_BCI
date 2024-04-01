
def frequencySelector(data ,freq: int):
    res = (freq - 8)/2
    return data[:, :, int(res), :]