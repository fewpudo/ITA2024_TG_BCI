from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import numpy as np
import matplotlib.pyplot as plt
import classifier
from data import plotAllChannelsFFTData, buildFFTData
import data


# Permitir escolher quais canais serão utilizados
# Permitir escolher quais frequências serão utilizadas


def trainningAcquisition(trainningTime, trial): #Retirar Trial
    # params = BrainFlowInputParams()
    # params.serial_port='/dev/tty.usbserial-DM00D434'
    # board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    try:
        # board.prepare_session()
        # board.start_stream()
        time.sleep(trainningTime)

        # data = board.get_board_data(sampling_rate*trainningTime) # data de cada janela
        # data_eeg = data[1:9,:]
        data_eeg = data.mockInputData(trial)
        
        # board.stop_stream()
        # board.release_session()
        return data_eeg
    except Exception as e:
        print("Ocorreu um erro durante a aquisição de dados:", str(e))
        # board.release_session()

def BCIOnline(trainningTime, w):
    params = BrainFlowInputParams()
    params.serial_port='/dev/tty.usbserial-DM00D434'
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    sampling_rate = 250
    try:
        board.prepare_session()
        board.start_stream()

        for i in range(trainningTime):
            time.sleep(1.5)
            data = board.get_board_data(sampling_rate)
            data_eeg = data[1:9,:]
            ypred = classifier.classify(data_eeg, w)

        board.stop_stream()
        board.release_session()
    except Exception as e:
        print("Ocorreu um erro durante a aquisição de dados:", str(e))
        board.release_session()