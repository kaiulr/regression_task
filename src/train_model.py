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
X = we[:, 1:] # Input matrix of all features
y = we[:, 0] # Output vector of fuel transmission


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

def Bias_Term(x): ## Offsets entire data by intercept (adding a column of 1s, similar to the first matrix in the PDF)
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1)) # Creating a new column of ones
    x=np.concatenate((b,x), axis=1) # Concatenating column to feature-matrix
    return x

def Predict(X,b):
    return (np.dot(X,b)) # Equation (43) from PDF

x_train=Bias_Term(X)
Beta=Train(x_train, y)
train_predict=Predict(x_train, Beta)

model_data = {
    'coefficients': Beta, 
    'features': ['Intercept'] + list(df.columns[1:])
} # Including intercept term here for completeness, though when running on actual model, re-initializing the intercept term as a column of 1s


## Creating a function for MSE, which we need for the final model metrics, and to act as a loss function to evaluation model performance

def metrics(predict, actual): # Calculating r2, MSE, RMSE from 2 arrays: predicted values and actual.
    tss = np.sum((predict - np.mean(actual)) ** 2)
    rss = np.sum((predict - actual) ** 2)
    r2 = 1 - (rss / tss)
    mse = np.mean((predict-actual) ** 2)
    return mse, np.sqrt(mse), r2


mse, rmse, r2 = metrics(y ,train_predict)

print('Training  Error for Multivariable regression using {} variables is {}  '.format(len(df.columns)-1, mse))

regression_metrics = ['Regression Metrics:\n', f'Mean Squared Error (MSE): {mse}\n', 
                      f'Root Mean Squared Error (RMSE): {rmse}\n', f'R-squared (R²) Score: {r2}\n']

with open(os.path.join(input_dir, 'results', 'metrics.txt'), 'w') as f:
    f.writelines(regression_metrics)

## Running cross-validation to see how the model performs across different subsets of data

# This function splits the data into n-folds, shuffles the folds, isolates one fold to act as the 'test' data
# Assuming fold of 5 as default value.

def create_folds(X, n=4):
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


def cross_validation(X, y, n_folds=4):
    folds = create_folds(X)
    mse_list = []
    rmse_list = []
    r2_list = []

    for i in range(n_folds-1):
        print(i)
        # Iteratively using ith fold as the test set, the rest as the training set
        test_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i]) # combining all remaining folds into one list
        if i>3:
            print(len(train_indices))


        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        

        X_train=Bias_Term(X_train)
        Beta=Train(X_train, y_train) # Getting coefficients

        X_test=Bias_Term(X_test)
        y_predict=Predict(X_test, Beta) # Using coefficients to predict on test dataset
        
        # Get metrics for test
        mse, rmse, r2 = metrics(y_test,y_predict)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
    
    # print('Mean MSE:{:%.2f}'.format(np.mean(mse_list)))
    print("oh")
    # print(f"Mean RMSE: {}", '{0:.2f}'.format(np.mean(rmse_list)))
    # print(f"Mean R2: {}", '{0:.2f}'.format(np.mean(r2_list)))


cross_validation(X, y)
## Visualizing errors
# plt.hist(errors_sqr, bins=100)
# plt.show()

## Visualizing Predicted vs. Actual Values
# plt.plot(xtrain, ytrain, 'o')
# plt.plot(xtrain, train_predict, 'x')
# plt.show()


## Saving model: coefficients corresponding to each feature used in model.
with open(os.path.join(input_dir, 'models', 'regression_model1.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to 'regression_model1.pkl'")