import numpy as np
import matplotlib.pyplot as plt
from utils import tools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Não preciso janelar, janelas 

# data chega num array de 8xAmostras
data = np.random.rand(8, 250)
samplingRate = 250
freqs = 4

def mockInputData():
    data = np.random.rand(8, 250)
    fftData = buildFFTData(data)
    return fftData.reshape(1,-1)


def carFilter(data):
    mean = np.mean(data, axis=0)
    car_data = data - mean.reshape(1,-1)
    return car_data

def standardizeData(data):
    cutoff_lowpass_freq = 5
    cutoff_notch_freq = 60
    Q = 30 
    lowpass_data = tools.butterLowpassFilter(data, cutoff_lowpass_freq, samplingRate)
    notch_data = tools.notchFilter(lowpass_data, cutoff_notch_freq, samplingRate, Q)
    res = carFilter(notch_data)
    return res

# ------------------------------------------------------------------------------------------------------------------------------------------
def fftTransform(windowData):
    fft_res = np.fft.fft(windowData)
    return fft_res

def buildFFTData(data):
    standardizedData = standardizeData(data)
    fftData = np.zeros((8, len(standardizedData[0])), dtype=object)
    res = np.zeros((8, freqs), dtype=object)
    for i in range(8):
        fftData[i] = abs(fftTransform(standardizedData[i,:]))
        res[i] = fftData[i][8], fftData[i][10], fftData[i][12], fftData[i][15]
    ones = np.ones((1,4), dtype=object)
    res = np.concatenate((res, ones), axis=0)
    return res #Nesse momento tenho a FFT da janela inteira


def plotFFTData(fft_res):
    frequencies = [8, 10, 12, 15]
    for j in range(8):
        plt.figure()
        plt.plot(np.abs(fft_res[j,:]))
        for freq in frequencies:
            plt.scatter(freq, np.abs(fft_res[j,freq]), color='red')
            plt.scatter(freq*2, np.abs(fft_res[j,freq*2]), color='red')
            plt.scatter(freq*3, np.abs(fft_res[j,freq*3]), color='red')
            plt.scatter(freq*4, np.abs(fft_res[j,freq*4]), color='red')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(5, 50)
        plt.title(f'FFT Data for Channel {j+1}')
        plt.show()


def plotAllChannelsFFTData(fft_res, i):
    frequencies = [8, 10, 12, 15]
    plt.figure()
    for j in range(8):
        plt.plot(np.abs(fft_res[j,:]), label=f'Channel {j+1}')
        for freq in frequencies:
            plt.scatter(freq, np.abs(fft_res[j,freq]), color='red')
            plt.scatter(freq*2, np.abs(fft_res[j,freq*2]), color='red')
            plt.scatter(freq*3, np.abs(fft_res[j,freq*3]), color='red')
            plt.scatter(freq*4, np.abs(fft_res[j,freq*4]), color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(5, 50)
    plt.title(f'FFT Data for All Channels window {i+1}')
    plt.legend()
    plt.show()


def buildNewLabelMatrix(evokedFreq):
    y = np.ones((9,freqs), dtype=object)*-1
    for i in range(8):
        y[i,evokedFreq-1] = 1
    return y

def createLinearClassifier(data, labels, x_entrada):
    # Prepare the data
    fftData = buildFFTData(data)
    print(fftData.shape)
    X = fftData.reshape(1,-1)  # Flatten the data
    y = labels.reshape(1,-1)  # Flatten the labels
    
    # Create and train the classifier
    pinvX = np.linalg.pinv(X.astype(np.float64))
    print(pinvX.shape, y.shape)
    W = np.matmul(pinvX,y)
    
    # Test the classifier
    y_pred = np.matmul(x_entrada,W)
    for i in range(y_pred.shape[0]):
        index = np.argmax(y_pred[i,:])
        y_pred[i,index] = 1
        y_pred[i,y_pred[i,:] != 1] = -1
    print(y_pred.shape)
    print(W.shape)
    return W

# Função para utilizar o classificador criado no treinamento
def classify(classifier, x_entrada):
    y_pred = np.matmul(x_entrada, classifier)
    for i in range(y_pred.shape[0]):
        index = np.argmax(y_pred[i,:])
        y_pred[i,index] = 1
        y_pred[i,y_pred[i,:] != 1] = -1
    return y_pred

# Example usage
x_teste = mockInputData()
labels = buildNewLabelMatrix(1)
classifier = createLinearClassifier(data, labels, x_teste) # Gera um classificador
print(classifier.shape)

print()
# Dúvidas sobre a online:
# Dimensão do W, o que significa cada linha?