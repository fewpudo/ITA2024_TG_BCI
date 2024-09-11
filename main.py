from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import numpy as np
import matplotlib.pyplot as plt
from data import DataMain

# Configurações da Cyton
params = BrainFlowInputParams()
params.serial_port = 'COM8'  # Substitua pelo seu serial port (em Windows pode ser COM3, por exemplo)

board = BoardShim(BoardIds.CYTON_BOARD.value, params)

# Inicialize a sessão
#board.release_session()
try:
    board.prepare_session()

    board.start_stream()
    print("Aquisitando dados...")

    data = board.get_board_data(1250)
    time.sleep(5)
    board.stop_stream()

    board.release_session()

    data_eeg = data[1:9,:]
    DataMain(data_eeg)

except:
    board.release_session()