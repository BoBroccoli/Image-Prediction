from numpy import *
import numpy as np

def sigmoid(x):
    return 1 / (1+exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNet:
    def __init__(self):
        self.hiddenLayerW = None
        self.outputLayerW = None
        self.output = None
        self.MSE = None
        self.trained = False
        
    def predict( self, X ):
        ### ... YOU FILL IN THIS CODE ....
        X = np.mat(X)
        X.reshape(256,1)
        L0 = np.hstack((np.array([[1] * X.shape[0]]).T, X))
        L0[:,0] = 1

        outputMatrix = np.asmatrix(self.outputLayerW)

        #print("+++++++++++")
        #print(L0.shape, self.hiddenLayerW.shape, outputMatrix.shape)

        h = sigmoid(np.dot(sigmoid(np.dot(L0,self.hiddenLayerW)),outputMatrix))
        return np.array(h)[0]

    def train(self,X,Y,hiddenLayerSize,epochs):    
        ## size of input layer (number of inputs plus bias)
        ni = X.shape[1] + 1

        ## size of hidden layer (number of hidden nodes plus bias)
        nh = hiddenLayerSize + 1

        # size of output layer
        no = 10

        ## initialize weight matrix for hidden layer
        self.hiddenLayerW = 2*random.random((ni,nh)) - 1

        ## initialize weight matrix for output layer
        self.outputLayerW = 2*random.random((nh,no)) - 1

        ## learning rate
        alpha = 0.001

        ## Mark as not trained 
        self.trained = False
        ## Set up MSE array
        self.MSE = [0]*epochs

        for epoch in range(epochs):

            ### ... YOU FILL IN THIS CODE ....
            a_0 = np.hstack((np.array([[1] * X.shape[0]]).T, X))
            in_0 = np.dot(a_0, self.hiddenLayerW)
            a_1 = sigmoid(in_0)

            a_1[:, 0] = 1
            in_1 = np.dot(a_1, self.outputLayerW)
            a_2 = sigmoid(in_1)
            error_out = Y - a_2
            delta_out = error_out * dsigmoid(a_2)
            ## Record MSE
            self.MSE[epoch] = mean(map(lambda x:x**2,error_out))

            ### ... YOU FILL IN THIS CODE
            error_hidden = np.dot(delta_out, self.outputLayerW.T)
            delta_hidden = error_hidden * dsigmoid(a_1)
            self.hiddenLayerW = self.hiddenLayerW + alpha * np.dot(a_0.T, delta_hidden)
            self.outputLayerW = self.outputLayerW + alpha * np.dot(a_1.T, delta_out)

        ## Update trained flag
        self.trained = True

