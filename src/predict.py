import pandas as pd
import numpy as np
import argparse
import pickle

# Since predict.py will parse user-inputted arguments, defining functions which take each command line argument as input

def load_model(model_path):
    with open(model_path, 'rb') as f:
        regression_model = pickle.load(f)
    return regression_model

def regression_results(df, output_path):
    df.to_csv(output_path, index=False)

# Function to save metrics
def save_metrics(train_error, r2, output_path):
    regression_metrics = ['Regression Metrics:\n', f'Mean Squared Error (MSE): {train_error}\n', 
                      f'Root Mean Squared Error (RMSE): {np.sqrt(train_error)}\n', f'R-squared (R²) Score: {r2}\n']
    with open(output_path, 'w') as f:
        f.writelines(regression_metrics)

# Function to calculate predictions using the coefficients
def predict(X, coefficients):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add the intercept term
    return X_b.dot(coefficients)

if __name__ == "__main__":
    
    # This function parses the command line arguments, in the exact format specified.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data (CSV file)')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path to save metrics')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path to save predictions')

    args = parser.parse_args()

    # Load saved model from path
    model_data = load_model(args.model_path)
    coefficients = model_data['coefficients']
    feature_names = model_data['features'][1:] 

    # Load data from the specified path
    data = pd.read_csv(args.data_path)
    
    # Ensure that the data has the correct features
    X = data[feature_names].values
    ytrain = data['FUEL CONSUMPTION'].values

    # Make predictions
    ypred = predict(X, coefficients)

    # Calculate metrics manually (MSE and R^2)
    mse = np.mean((ypred - ytrain) ** 2)
    
    # Total sum of squares (TSS) and residual sum of squares (RSS)
    tss = np.sum((ytrain - np.mean(ytrain)) ** 2)
    rss = np.sum((ypred - ytrain) ** 2)
    r2 = 1 - (rss / tss)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': ytrain,
        'Predicted': ypred
    })

    regression_results(predictions_df, args.predictions_output_path)

    # Save metrics to a text file
    save_metrics(mse, r2, args.metrics_output_path)

    print(f"Predictions saved to {args.predictions_output_path}")
    print(f"Metrics saved to {args.metrics_output_path}")
