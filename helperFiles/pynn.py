import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from .logger import *

# SKLEARN: solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1


###
# Basic configuration taken from link below and modified
#  https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
#
###
def runPyNN(X_train, X_test, y_train):
#    X = torch.FloatTensor(X_train)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y = torch.tensor(y_train)

    #Initialize the model        
    model = Net(X_train.shape[1])
    #Define loss criterion
    criterion = nn.CrossEntropyLoss()
    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)     # errors...
    
    #Number of epochs
    epochs = 10000
    #List to store losses
    losses = []
    for i in range(epochs):
        #Precit the output for Given input
        y_pred = model.forward(X_train)
        #Compute Cross entropy loss
        loss = criterion(y_pred,y)
        #Add loss to the list
        losses.append(loss.item())
        #Clear the previous gradients
        optimizer.zero_grad()
        #Compute gradients
        loss.backward()
        #Adjust weights
        optimizer.step()

    return model.predict(X_test).numpy()

#our class must extend nn.Module
class Net(nn.Module):
    def __init__(self, inputLayer):
        super(Net,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(inputLayer,100)
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(100,2)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. 
        x = F.tanh(x)
        #This produces output
        x = self.fc2(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

