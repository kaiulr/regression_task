import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from data_preprocessing import preprocessing
import pickle

## Beginning with OLS - Multivariate Regression
# Based on results from data exploration, using all numerical IVs (Coemissions, Engine Size, Cylinders)
# Also using encoded Fuel and Transmission values as they are not highly branched.

input_file = "C:/Users/Fiona/Desktop/Fiona_Arora_A1/regression_task/fuel_train.csv"
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
data_dir = preprocessing(input_file, output_dir)
df = pd.read_csv(data_dir)

# One data pre-processed, selecting which columns to keep for regression task
fuel_dummy = [column for column in df.columns if 'FUEL_' in column]
transmission_dummy = [column for column in df.columns if 'TRANSMISSION' in column]
# keep_columns = ['COEMISSIONS','ENGINE SIZE', 'CYLINDERS'] + transmission_dummy + fuel_dummy
keep_columns = ['scaled_coemissions','ENGINE SIZE', 'CYLINDERS'] + transmission_dummy + fuel_dummy

# df = df[keep_columns]
# Converting DF to Numpy array to perform matrix operations
# we=df.to_numpy()
# print(we)
# we=we.astype(np.float64)
# X = we[:, 1:] # Input matrix of all features
# print(X)
# y = we[:, 0] # Output vector of fuel transmission

X = df[keep_columns].values
y = df['FUEL CONSUMPTION'].values

## Find the OLS estimators for an input matrix X and singular numeric output Y
# Using Equation 12 from: https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
def Train(X,Y, lmbda= 0.05):
    X = X.astype(float)
    first = np.dot(X.T, X)
    identity_matrix = np.eye(X.shape[1])
    
    first_reg = first + lmbda * identity_matrix
    
    inverse = np.linalg.inv(first_reg)  # Invert the regularized matrix
    second = np.dot(X.T, Y)
    
    b = np.dot(inverse, second)
    return b

def Bias_Term(x): ## Offsets entire data by intercept (adding a column of 1s, similar to the first matrix in the PDF)
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1)) # Creating a new column of ones
    x=np.concatenate((b,x), axis=1) # Concatenating column to feature-matrix
    return x

def Predict(X,b):
    return (np.dot(X,b)) # Equation (43) from PDF (multiplying features with coefficients)

# Taking train-test split (80:20)
split_index = int(np.ceil(len(X)*0.8))

trainx = X[0:split_index]
trainy = y[0:split_index]
testx = X[split_index + 1:]
testy = y[split_index + 1:]

x_train=Bias_Term(trainx)
Beta=Train(x_train, trainy)

x_test = Bias_Term(testx)
test_predict=Predict(x_test, Beta)

## Visualizing Predicted vs. Actual Values
plt.plot(testy, test_predict, 'x')
plt.title("Plotting Actual vs Predicted Fuel Consumption Values on Test Set")
plt.show()


def metrics(actual, predicted): # Calculating r2, MSE, RMSE from 2 arrays: predicted values and actual.
    tss = np.sum((actual - np.mean(actual)) ** 2)
    ssr = np.sum((predicted - actual)**2)
    r2 = 1 - (ssr / tss)
    mse = np.mean((predicted-actual) ** 2)
    return mse, np.sqrt(mse), r2

mse, rmse, r2 = metrics(testy, test_predict)

print('Test Error for Multivariate Ridge Regression using {} variables is {}  '.format(len(df.columns)-1, round(mse,3)))


## Running cross-validation to see how the model performs across different subsets of data

# This function splits the data into n-folds, shuffles the folds, isolates one fold to act as the 'test' data
# Assuming fold of 10 as default value.

def create_folds(X, n):
    # Gets us evenly spaced index for the entire length of the input matrix X
    # Equal to the number of data points.
    indices = np.arange(len(X))

    np.random.shuffle(indices) # Using in-built numpy library to shuffle the row indices, in order to create randomly ordered folds
    
    interval_size = len(X) // n # How many data points we want in each fold

    # Create folds using shuffled indices
    # List of lists, where each sub-list is fold_size long
    # Slicing 'indices' into 1st fold, then jumping ahead 1*fold_size points to get next fold, etc.
    n_folds = [indices[i * interval_size:(i + 1) * interval_size] for i in range(n)]
    
    # Add left over datapoints to the last fold (n_folds[-1])
    # Remaining points are all those that are present after the last n*interval_size index.
    if len(X) % n != 0:
        n_folds[-1] = np.concatenate([n_folds[-1], indices[n * interval_size:]])

    return n_folds


def cross_validation(X, y, n_folds, lmbda=0.01):
    folds = create_folds(X, n_folds)
    mse_list = []
    rmse_list = []
    r2_list = []
    beta_list = []

    for i in range(n_folds):
        # print(f"Using fold {i} as Test Set.")
        # Iteratively using ith fold as the test set, the rest as the training set
        test_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i]) # combining all remaining folds into one list

        # print(list(set(train_indices) & set(test_indices))) # Checking for any overlap in train and test set

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # print(f"Train Set Size: {len(train_indices)}, Test Set Size: {len(test_indices)}")

        X_train=Bias_Term(X_train)
        Beta=Train(X_train, y_train, lmbda=lmbda) # Getting coefficients from training data

        X_test=Bias_Term(X_test)
        y_predict=Predict(X_test, Beta) # Using coefficients to predict on test dataset
        
        # Get metrics for test
        mse, rmse, r2 = metrics(y_test, y_predict)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        beta_list.append(Beta)
    
    # print("Mean MSE:", round(np.mean(mse_list), 2))
    # print("Mean RMSE:", round(np.mean(rmse_list), 2))
    # print("Mean R2:", round(np.mean(r2_list), 2))
    return mse_list, rmse_list, r2_list, beta_list

# n_folds = 4
# # mse, rmse, r2, betas = cross_validation(X, y, n_folds)
# print(mse)
# ## Visualizing performance with each fold
# x_axis = [f"Fold {i}" for i in range(1,n_folds+1)]
# def addlabels(x,y):
#     for i in range(len(x)):
#         plt.text(i,y[i],format(y[i], ".2f"))

# plt.plot(x_axis, mse, label="Mean Squared Error", marker='o')
# addlabels(x_axis, mse)
# plt.plot(x_axis, r2, label="R-Squared", marker='o')
# addlabels(x_axis, r2)
# plt.legend()
# plt.title("Fold-Wise Performance")
# plt.show()

## Averaging the betas from cross validation
# stacked_arrays = np.vstack(betas)
# averaged_betas = np.mean(stacked_arrays, axis=0)
# averaged_coeff = averaged_betas.tolist()

# X_final = Bias_Term(X)
# averaged_predict = Predict(X_final, averaged_coeff)

# print(metrics(y, averaged_predict))

for i in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
    n_folds = 5
    mse,_,r2,_ = cross_validation(X, y, n_folds, lmbda=i)
    print(f"Average Test MSE across {n_folds} folds for Lambda={i}: {round(np.mean(mse),3)}")
    print(f"Average Test R2 across {n_folds} folds for Lambda={i}: {round(np.mean(r2),3)}")


model1_data = {
    'coefficients': Beta, 
    'features': list(keep_columns)
}

## Saving model: coefficients corresponding to each feature used in model.
with open(os.path.join(model_dir, 'regression_model1.pkl'), 'wb') as f:
    pickle.dump(model1_data, f)

print("Model saved to 'regression_model1.pkl'")

# model2_data = {
#     'coefficients': averaged_coeff, 
#     'features': list(keep_columns)
# }

# with open(os.path.join(model_dir, 'regression_model2.pkl'), 'wb') as f:
#     pickle.dump(model2_data, f)

# print("Model saved to 'regression_model2.pkl'")

