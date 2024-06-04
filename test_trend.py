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
predictions = []
for i in range(n_steps, len(data)):
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

# Plot the trend of one feature along with predictions
plt.figure(figsize=(15, 5))
plt.plot(data[:, 0], label='Actual Data')
plt.plot(np.arange(n_steps, len(data)), predictions[:, 0], label='Predicted Data', linestyle='dashed')
plt.plot(smoothed_data[:, 0], label='Smoothed Data', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Feature 1 Value')
plt.title('Trend Analysis with LSTM Predictions')
plt.legend()
plt.show()
