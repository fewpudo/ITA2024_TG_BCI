from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import numpy as np
import matplotlib.pyplot as plt
from data import plotAllChannelsFFTData, buildFFTData

params = BrainFlowInputParams()
params.serial_port='/dev/tty.usbserial-DM00D434'
board = BoardShim(BoardIds.CYTON_BOARD.value, params)

try:
    board.prepare_session()
    
    board.start_stream()

    for i in range(7):
        print(f"Iteração número {i}")
        data = board.get_board_data(250)
        print("Conseguiu pegar os dados")
        time.sleep(1.5)
        data_eeg = data[1:9,:]
        plt.ion()
        plotAllChannelsFFTData(buildFFTData(data_eeg),i)
    plt.ioff()
    plt.show()

    board.stop_stream()
    board.release_session()
except Exception as e:
    print("Ocorreu um erro durante a aquisição de dados:", str(e))
    board.release_session()