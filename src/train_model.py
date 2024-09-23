import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


## Beginning with OLS - Multivariable Regression
# Based on results from data exploration, using all numerical IVs (Coemissions, Engine Size, Cylinders)
# Also using encoded Fuel and Transmission values as they are not highly branched.

input_dir = os.path.join(os.path.dirname((os.path.abspath(""))), 'data') 
df = pd.read_csv(os.path.join(input_dir, 'training_data.csv'))
fuel_dummy = [column for column in df.columns if 'FUEL' in column]
transmission_dummy = [column for column in df.columns if 'TRANSMISSION' in column]
keep_columns = ['FUEL CONSUMPTION', 'COEMISSIONS','ENGINE SIZE', 'CYLINDERS'] + transmission_dummy + fuel_dummy
# print(keep_columns)
df = df[keep_columns]

# Converting DF to Numpy array to perform matrix operations
we=df.to_numpy()
we=we.astype(np.float64)
xtrain = we[:, 1:5] # Input matrix of all 5 features
ytrain = we[:, 0] # Output vector of fuel transmission

## Creating a function for MSE, which we need for the final model metrics, and to act as a loss function to evaluation model performance
# Formula is simply 1/n * sum^n_{i=1} x_i - x_hat, or 1/n * SSE
def mse(actual, predicted):
    # sum_error = 0.0 # Initialize SSE to 0
    # for x in range(len(actual)):
    #     error = predicted[x] - actual[x]
    #     sum_error += (error ** 2)
    #     mean_error = sum_error / float(len(actual))/
    mean_error = np.mean((predicted-actual) ** 2)
    return mean_error


## Find the OLS estimators for an input matrix X and singular numeric output Y
# Using Equation 12 from: https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
def Train(X,Y):
    X.astype(float)
    first=np.dot(X.T,X)
    first.astype(np.float16)
    inverse=np.linalg.inv(first)
    second=np.dot(X.T,Y)
    
    b=np.dot(inverse,second)
    return b

def Bias_Term(x):
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1))
    x=np.concatenate((b,x), axis=1)
    return x

def Predict(X,b):
    return (np.dot(X,b))

for i in range(2,6):
    x_train=Bias_Term(xtrain[:,0:i])
    b=Train(x_train,ytrain)
    train_predict=Predict(x_train,b)
    train_error=mse(ytrain,train_predict)
    print('Training  Error for Multivariable regression using  {} variables is   {}  '.format(i,train_error))

errors = train_predict - ytrain
errors_sqr = errors**2

tss = np.sum((train_predict - np.mean(ytrain)) ** 2)
rss = np.sum((train_predict - ytrain) ** 2)
r2 = 1 - (rss / tss)
print(r2)

# plt.hist(errors_sqr, bins=100)
# plt.show()

plt.plot(xtrain, ytrain, 'o')
#add linear regression line to scatterplot 
plt.plot(xtrain, train_predict, 'x')
plt.show()
