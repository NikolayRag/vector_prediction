from _plotlib import *

import matplotlib.pyplot as plt
import numpy as np
from vector_prediction import VectorSignalPredictor

# Initialize the predictor
n_steps = 20
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Example data
data = np.random.rand(30, 3)  # 50 time steps, 3 features

# Train the model
predictor.fit(data, epochs=20, split_ratio=0.8)

# Analyze trend by making continuous predictions
predictions = [np.array([[0,0,0]])]*(len(data)-n_steps)
for i in range(len(data)-n_steps, len(data)):
    print(f"Prediction {i}/{len(data)}")

    X_new = data[i-n_steps:i]
    y_pred, _ = predictor.predict(X_new, n_iter=10)
    predictions.append(y_pred)

# Convert to array for easier manipulation
predictions = np.array(predictions).squeeze()

# Smoothing the data
smoothed_data = predictor.smooth_data_ema(data, alpha=0.1)

# Prediction Uncertainty
_, uncertainty = predictor.predict(data[-n_steps:], n_iter=100)
print("Prediction uncertainty for latest prediction (standard deviation):", uncertainty)

print(np.arange(n_steps, len(data)))
plot_start()
plot_vectors_layer(pd.DataFrame(data[:, 0]), label='Actual Data')
plot_vectors_layer(pd.DataFrame(predictions[:, 0]), label='Predicted Data', linestyle='dashed')
plot_vectors_layer(pd.DataFrame(smoothed_data[:, 0]), label='Smoothed Data', linestyle='dashed')
plot_end()
