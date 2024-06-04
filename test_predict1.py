'''
Test: learn on data and predict one step more.
'''

import pandas as pd
import matplotlib.pyplot as plt

def plot_vectors(data):
    # Create a new figure for plotting
    plt.figure()
    
    # Plot each column as a separate line
    for column in data.columns:
        plt.plot(data[column], label=f'Line {column + 1}')
        
    # Add title and labels
    plt.title('Plot of Vectors from CSV')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

#pData = plot_vectors(pd.read_csv('fft.csv', header=None)) # Plot .csv file for example, where row is vector



import numpy as np
from vector_prediction import VectorSignalPredictor

# Initialize the predictor
n_steps = 20  # Number of previous steps to use for prediction
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Generate some synthetic data for demonstration purposes
data = np.random.rand(1000, 6)  # 1000 time steps, each with a 6-dimensional vector

# Train the model
predictor.fit(data, epochs=20, split_ratio=0.8)

factor = .01
data = predictor.smooth_data_ema(data, factor/(factor+1.0))

# Predict the next step
X_new = data[-n_steps:]  # Take the last n_steps vectors
y_pred, uncertainty = predictor.predict(X_new, 100)

# Print prediction and uncertainty
print(X_new)
print("Predicted next vector:", y_pred)
print("Prediction uncertainty (standard deviation):", uncertainty)

# Calculate and print expected error
expected_error = predictor.expected_error(data)
print("Expected Mean Absolute Error on Validation Set:", expected_error)


plot_vectors(pd.DataFrame(X_new))

