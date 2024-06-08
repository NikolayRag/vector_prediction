'''
Cumulative prediction test
'''
   
import _fixpath
from _jsonadd import *
from _plotlib import *

import matplotlib.pyplot as plt
import numpy as np
from vector_prediction import VectorSignalPredictor

# Example data
data = load_json()
if data is None:
    data = np.random.rand(1000, 9)  # 1000 time steps, 9 features
print(f"Input data of {len(data)} x {len(data[0])}")

# Initialize the predictor
n_steps = 50
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Train the model
epochs=100
predictor.fit(data, epochs=epochs, split_ratio=0.8)

# Analyze trend by making continuous cumulative predictions
goback = 50
p_iter=25
predictions, _ = predictor.predict_some(data[:-goback], steps=goback, n_iter=p_iter)

# Smoothing the data
smoothed_data = predictor.smooth_data_ema(data, alpha=0.1)

plot_start()
plot_vectors_layer(pd.DataFrame(data[-200:,0]), label='Actual Data')
plot_vectors_layer(pd.DataFrame((np.append(data[:-goback],predictions,axis=0))[-200:,0]), label='Predicted Data', format=':')
plot_vectors_layer(pd.DataFrame(smoothed_data[-200:,0]), label='Smoothed Data', format='--')
plot_end()

