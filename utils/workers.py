import numpy as np

# verify if the data is correct
# This function gets the data randomly and separates it into validation and test data 
def buildValidationAndTestMatrix(data):
    validationMatrix = data[np.random.choice(data.shape[0], 24, replace=False), :]
    testMatrix = data[np.random.choice(np.setdiff1d(np.arange(data.shape[0]), validationMatrix[:, 0]), 96, replace=False), :]
    return validationMatrix, testMatrix


def buildY8Matrix():
    y_matrix = np.ones((120, 1))
    y_matrix[:30] = 1
    y_matrix[30:60] = -1
    y_matrix[60:90] = -1
    y_matrix[90:] = -1
    return y_matrix

def buildY10Matrix():
    y_matrix = np.ones((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = 1
    y_matrix[60:90] = -1
    y_matrix[90:] = -1
    return y_matrix

def buildY12Matrix():
    y_matrix = np.ones((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = -1
    y_matrix[60:90] = 1
    y_matrix[90:] = -1
    return y_matrix

def buildY15Matrix():
    y_matrix = np.ones((120, 1))
    y_matrix[:30] = -1
    y_matrix[30:60] = -1
    y_matrix[60:90] = -1
    y_matrix[90:] = 1
    return y_matrix

def buildYMatrix():
    y8 = buildY8Matrix()
    y10 = buildY10Matrix()
    y12 = buildY12Matrix()
    y15 = buildY15Matrix()
    return y8, y10, y12, y15