'''
Test: apply different smoothing effects to find best uncertainity.
'''

import _fixpath
from _jsonadd import *

import numpy as np
from vector_prediction import VectorSignalPredictor



def analyze_smoothing_effectsSMA(data, n_steps, smoothing_factors, epochs=20):
    predictor = VectorSignalPredictor(n_steps)

    best_factor = None
    best_uncertainty = float('inf')

    for factor in smoothing_factors:
        print(f"Analyzing with smoothing factor: {factor}")

        # Apply SMA smoothing
        data_sma = predictor.smooth_data_sma(data, factor)
        predictor.fit(data_sma, epochs)
        X_new_sma = data_sma[-n_steps:]
        _, uncertainty_sma = predictor.predict(X_new_sma)
        mean_uncertainty_sma = np.mean(uncertainty_sma)

        # Compare uncertainties and choose the best smoothing method
        if mean_uncertainty_sma < best_uncertainty:
            best_uncertainty = mean_uncertainty_sma
            best_factor = ('SMA', factor)

    return best_factor, best_uncertainty


def analyze_smoothing_effectsEMA(data, n_steps, smoothing_factors, epochs=20):
    predictor = VectorSignalPredictor(n_steps)

    best_factor = None
    best_uncertainty = float('inf')

    for factor in smoothing_factors:
        print(f"Analyzing with smoothing factor: {factor}")

        # Apply EMA smoothing
        data_ema = predictor.smooth_data_ema(data, factor/(factor+1.0))
        predictor.fit(data_ema, epochs)
        X_new_ema = data_ema[-n_steps:]
        _, uncertainty_ema = predictor.predict(X_new_ema)
        mean_uncertainty_ema = np.mean(uncertainty_ema)
        
        
        if mean_uncertainty_ema < best_uncertainty:
            best_uncertainty = mean_uncertainty_ema
            best_factor = ('EMA', factor)

    return best_factor, best_uncertainty


# Load or Generate some sample data
data = load_json()
if data is None:
    data = np.random.rand(1000, 3)  # 1000 time steps with 3 features
print(f"Input data of {len(data)} x {len(data[0])}")

n_steps = 10  # Number of previous steps to use for prediction
epochs = 20  # Number of training epochs
smoothing_factors = [5, 10, 20]  # Different smoothing factors to analyze

print("SMA")
best_factorSMA, best_uncertaintySMA = analyze_smoothing_effectsSMA(data, n_steps, smoothing_factors, epochs)

print("EMA")
best_factorEMA, best_uncertaintyEMA = analyze_smoothing_effectsEMA(data, n_steps, smoothing_factors, epochs)

print(f"Best simple smoothing factor : {best_factorSMA} with uncertainty: {best_uncertaintySMA}")
print(f"Best exponental smoothing factor: {best_factorEMA} with uncertainty: {best_uncertaintyEMA}")
