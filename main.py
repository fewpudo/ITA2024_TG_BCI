import csv
import numpy as np
import scipy
from scipy.signal import butter, filtfilt


def readData(data):
    read = scipy.io.loadmat('FILENAME')
    #Dados de uma varíavel específica : var_data = read['VAR_NAME']
    return read

def choicePathsByFreq(data,freqs):
    #depende do tipo do arquivo
    return res