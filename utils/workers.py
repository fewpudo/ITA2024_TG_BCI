import numpy as np

# verify if the data is correct
# This function gets the data randomly and separates it into validation and test data 
def buildValidationAndTestMatrix(data):
    validationMatrix = data[np.random.choice(data.shape[0], 24, replace=False), :]
    testMatrix = data[np.random.choice(np.setdiff1d(np.arange(data.shape[0]), validationMatrix[:, 0]), 96, replace=False), :]
    return validationMatrix, testMatrix