from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import numpy as np
import matplotlib.pyplot as plt
from data import plotAllChannelsFFTData, buildFFTData
from openbci import cyton as bci

# Configurações da Cyton

board = bci.OpenBCICyton(daisy=False)


# Inicialize a sessão
try:
    board.prepare_session()

    board.start_stream()
    for i in range(7):
        data = board.get_board_data(250)
        time.sleep(1.5)
        data_eeg = data[1:9,:]
        plotAllChannelsFFTData(buildFFTData(data_eeg))


    board.stop_stream()
    board.release_session()
except Exception as e:
    print("Ocorreu um erro durante a aquisição de dados:", str(e))
    board.release_session()