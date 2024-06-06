'''
Test: learn on data and predict one step more.
'''


import _fixpath
from _jsonadd import *
from _plotlib import *

import numpy as np
from vector_prediction import VectorSignalPredictor

# Initialize the predictor
n_steps = 50  # Number of previous steps to use for prediction
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Load or Generate some synthetic data for demonstration purposes
data = load_json()
if data is None:
    data = np.random.rand(1000, 6)  # 1000 time steps, each with a 6-dimensional vector
print(f"Input data of {len(data)} x {len(data[0])}")

# Train the model
predictor.fit(data, epochs=20, split_ratio=0.8)

#factor = .01
#data = predictor.smooth_data_ema(data, factor/(factor+1.0))

# Predict the next step
X_new = data[-n_steps:]  # Take the last n_steps vectors
y_pred, uncertainty = predictor.predict(X_new, 100)

# Print prediction and uncertainty
print(X_new[-3:])
print("Predicted next vector:", y_pred)
print("Prediction uncertainty (standard deviation):", uncertainty)

# Calculate and print expected error
expected_error = predictor.expected_error(data)
print("Expected Mean Absolute Error on Validation Set:", expected_error)


plot_start()

toplot = np.append(data, y_pred, axis=0)
plot_vectors_layer(pd.DataFrame(toplot[-n_steps-1:]))

for y in y_pred[0]:
    plot_vectors_points(n_steps, y, 'o')


plot_end()
