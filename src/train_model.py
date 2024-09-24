import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

## Beginning with OLS - Multivariable Regression
# Based on results from data exploration, using all numerical IVs (Coemissions, Engine Size, Cylinders)
# Also using encoded Fuel and Transmission values as they are not highly branched.

input_dir = os.path.join(os.path.dirname((os.path.abspath("")))) 
df = pd.read_csv(os.path.join(input_dir, 'data', 'training_data.csv'))
fuel_dummy = [column for column in df.columns if 'FUEL' in column]
transmission_dummy = [column for column in df.columns if 'TRANSMISSION' in column]
keep_columns = ['FUEL CONSUMPTION', 'COEMISSIONS','ENGINE SIZE', 'CYLINDERS'] + transmission_dummy + fuel_dummy

df = df[keep_columns]

# Converting DF to Numpy array to perform matrix operations
we=df.to_numpy()
we=we.astype(np.float64)
xtrain = we[:, 1:] # Input matrix of all features
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

# for i in range(2,6):
#     x_train=Bias_Term(xtrain[:,0:i])
#     b=Train(x_train,ytrain)
#     train_predict=Predict(x_train,b)
#     train_error=mse(ytrain,train_predict)
#     print('Training  Error for Multivariable regression using  {} variables is   {}  '.format(i,train_error))

x_train=Bias_Term(xtrain)
Beta=Train(x_train,ytrain)
train_predict=Predict(x_train, Beta)
train_error=mse(ytrain,train_predict)
print('Training  Error for Multivariable regression using {} variables is {}  '.format(len(df.columns)-1,train_error))
print(Beta)

model_data = {
    'coefficients': Beta, 
    'features': ['Intercept'] + list(df.columns[1:])
}



tss = np.sum((train_predict - np.mean(ytrain)) ** 2)
rss = np.sum((train_predict - ytrain) ** 2)
r2 = 1 - (rss / tss)
print(r2)

regression_metrics = ['Regression Metrics:\n', f'Mean Squared Error (MSE): {train_error}\n', 
                      f'Root Mean Squared Error (RMSE): {np.sqrt(train_error)}\n', f'R-squared (R²) Score: {r2}\n']

with open(os.path.join(input_dir, 'results', 'metrics.txt'), 'w') as f:
    f.writelines(regression_metrics)

errors = train_predict - ytrain
errors_sqr = errors**2
## Visualizing errors
# plt.hist(errors_sqr, bins=100)
# plt.show()

## Visualizing Predicted vs. Actual Values
# plt.plot(xtrain, ytrain, 'o')
# plt.plot(xtrain, train_predict, 'x')
# plt.show()

with open(os.path.join(input_dir, 'models', 'regression_model1.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to 'regression_model1.pkl'")