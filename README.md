# Vector Signal Prediction Library

The `VectorSignalPredictor` library provides tools to predict future values of vector signals using LSTM neural networks and measure the associated prediction uncertainty.

## Installation

To use this library, simply clone the repository and import it into your project.

```
$ git clone <repository-url>
```

## Usage

### Importing the Library

First, import the library and create an instance of `VectorSignalPredictor`.

```python
from vector_signal_prediction import VectorSignalPredictor
```

### Initializing the Predictor

Create an instance of the `VectorSignalPredictor` class:

```python
n_steps = 10  # Number of previous steps to use for prediction
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)
```

### Training the Model

Train the LSTM model on your data:

```python
data = np.random.rand(1000, 3)  # 1000 time steps, each with a 3-dimensional vector
predictor.fit(data, epochs=20, split_ratio=0.8)
```

### Making Predictions

To make a prediction for the next time step and measure the uncertainty:

```python
X_new = data[-n_steps:]
y_pred, uncertainty = predictor.predict(X_new)

print("Predicted next vector:", y_pred)
print("Prediction uncertainty (standard deviation):", uncertainty)
```

### Checking Expected Error

Calculate the expected error on the validation set:

```python
expected_error = predictor.expected_error(data)
print("Expected Mean Absolute Error on Validation Set:", expected_error)
```

# VectorSignalPredictor Library reference

The `VectorSignalPredictor` library provides an easy-to-use interface for building and training LSTM models for vector signal prediction. It supports preprocessing, training, and prediction with uncertainty estimation.

## Class `VectorSignalPredictor`

### Constructor

`__init__(self, n_steps: int, dropout_rate: float = 0.2)`

- **Parameters:**
  - `n_steps` (int): Number of previous steps to use for prediction.
  - `dropout_rate` (float): Dropout rate for regularization (default is 0.2).

### Methods

#### `preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Preprocess the input data into sequences.

- **Parameters:**
  - `data` (np.ndarray): Input data with shape (timesteps, features).

- **Returns:**
  - `X` (np.ndarray): Preprocessed input data.
  - `y` (np.ndarray): Preprocessed output data.

#### `prepare_dataset(self, data: np.ndarray, split_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

Prepare the dataset for training and validation.

- **Parameters:**
  - `data` (np.ndarray): Input data.
  - `split_ratio` (float): Ratio for splitting the data into training and validation sets (default is 0.8).

- **Returns:**
  - `X_train` (np.ndarray): Training input data.
  - `y_train` (np.ndarray): Training target data.
  - `X_val` (np.ndarray): Validation input data.
  - `y_val` (np.ndarray): Validation target data.

#### `build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential`

Build the LSTM model.

- **Parameters:**
  - `input_shape` (tuple): Shape of the input to the LSTM model.

- **Returns:**
  - `model` (Sequential): Compiled LSTM model.

#### `fit(self, data: np.ndarray, epochs: int = 20, split_ratio: float = 0.8) -> None`

Train the LSTM model.

- **Parameters:**
  - `data` (np.ndarray): Input data.
  - `epochs` (int): Number of epochs to train (default is 20).
  - `split_ratio` (float): Ratio for splitting the data into training and validation sets (default is 0.8).

#### `predict_with_uncertainty(self, X: np.ndarray, n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]`

Predict with uncertainty using multiple forward passes.

- **Parameters:**
  - `X` (np.ndarray): Input data for prediction.
  - `n_iter` (int): Number of stochastic forward passes (default is 50).

- **Returns:**
  - `prediction` (np.ndarray): Mean prediction.
  - `uncertainty` (np.ndarray): Prediction uncertainty (standard deviation).

#### `predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Make a prediction for the next time step.

- **Parameters:**
  - `X` (np.ndarray): Input data for prediction.

- **Returns:**
  - `y_pred` (np.ndarray): Predicted next vector.
  - `uncertainty` (np.ndarray): Prediction uncertainty.

#### `expected_error(self, data: np.ndarray) -> float`

Calculate the expected error on the validation set.

- **Parameters:**
  - `data` (np.ndarray): Input data.

- **Returns:**
  - `error` (float): Expected Mean Absolute Error (MAE).

#### `smooth_data_sma(self, data, window_size) -> np.ndarray`

Applies a Simple Moving Average (SMA) smoothing to the given data over a specified window size.

- **Parameters:**
  - `data` (np.ndarray): The input data array with shape `(timesteps, features)`.
  - `window_size` (int): The size of the window over which to calculate the moving average.

- **Returns:**
  - `smoothed_data` (np.ndarray): The data array after applying the simple moving average smoothing. The array has the same shape as the input data.

#### `smooth_data_ema(self, data, alpha) -> np.ndarray`

Applies an Exponential Moving Average (EMA) smoothing to the given data with a specified smoothing factor `alpha`.

- **Parameters:**
  - `data` (np.ndarray): The input data array with shape `(timesteps, features)`.
  - `alpha` (float): The smoothing factor for the exponential moving average. It determines the weight of more recent observations compared to older ones. The value of `alpha` should be between 0 and 1.

- **Returns:**
  - `ema_data` (np.ndarray): The data array after applying the exponential moving average smoothing. The array has the same shape as the input data.

### Example Code Usage

```python
import numpy as np
from prediction import VectorSignalPredictor

# Initialize the predictor
n_steps = 10  # Number of previous steps to use for prediction
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Generate some synthetic data for demonstration purposes
data = np.random.rand(1000, 3)  # 1000 time steps, each with a 3-dimensional vector

# Train the model
predictor.fit(data, epochs=20, split_ratio=0.8)

# Predict the next step
X_new = data[-n_steps:]  # Take the last n_steps vectors
y_pred, uncertainty = predictor.predict(X_new)

# Print prediction and uncertainty
print("Predicted next vector:", y_pred)
print("Prediction uncertainty (standard deviation):", uncertainty)

# Calculate and print expected error
expected_error = predictor.expected_error(data)
print("Expected Mean Absolute Error on Validation Set:", expected_error)
```
