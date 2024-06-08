import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

class VectorSignalPredictor:
    def __init__(self, n_steps, dropout_rate=0.2):
        """
        Initialize the predictor with the given number of time steps and dropout rate.
        
        Args:
            n_steps (int): Number of previous steps to use for prediction.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.n_steps = n_steps
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler()
        self.model = None

    def preprocess_data(self, data):
        """
        Preprocess the input data into sequences.
        
        Args:
            data (np.ndarray): Input data with shape (timesteps, features).
        
        Returns:
            X (np.ndarray): Preprocessed input data.
            y (np.ndarray): Preprocessed output data.
        """
        X, y = [], []
        for i in range(len(data) - self.n_steps):
            X.append(data[i:i+self.n_steps])
            y.append(data[i+self.n_steps])
        return np.array(X), np.array(y)

    def prepare_dataset(self, data, split_ratio=0.8):
        """
        Prepare the dataset for training and validation.
        
        Args:
            data (np.ndarray): Input data.
            split_ratio (float): Ratio for splitting the data into training and validation sets.
        
        Returns:
            tuple: Train and validation sets along with scaler object.
        """
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = self.preprocess_data(data_scaled)
        
        split_index = int(split_ratio * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        return X_train, y_train, X_val, y_val

    def custom_loss_function(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred)) + tf.reduce_mean(tf.square(tf.reduce_sum(y_true, axis=-1) - tf.reduce_sum(y_pred, axis=-1)))


    def build_lstm_model(self, input_shape):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(50, activation='relu', return_sequences=True)(inputs)
        attention_out = Attention()([lstm_out, lstm_out])
        lstm_out2 = LSTM(50, activation='relu')(attention_out)
        outputs = Dense(input_shape[1])(lstm_out2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=self.custom_loss_function)
        return model

    '''
    def build_lstm_model(self, input_shape):
        """
        Build the LSTM model.
        
        Args:
            input_shape (tuple): Shape of the input to the LSTM model.
        
        Returns:
            model (Sequential): Compiled LSTM model.
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(50, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(input_shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    '''

    def fit(self, data, epochs=20, split_ratio=0.8):
        """
        Train the LSTM model.
        
        Args:
            data (np.ndarray): Input data.
            epochs (int): Number of epochs to train.
            split_ratio (float): Ratio for splitting the data into training and validation sets.
        """
        X_train, y_train, X_val, y_val = self.prepare_dataset(data, split_ratio)
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

    def predict_with_uncertainty(self, X, n_iter=50):
        """
        Predict with uncertainty using multiple forward passes.
        
        Args:
            X (np.ndarray): Input data for prediction.
            n_iter (int): Number of stochastic forward passes.
        
        Returns:
            tuple: Mean prediction and uncertainty (standard deviation).
        """
        preds = np.zeros((n_iter, X.shape[0], X.shape[2]))
        for i in range(n_iter):
            preds[i, :, :] = self.model(X, training=True)
        prediction = preds.mean(axis=0)
        uncertainty = preds.std(axis=0)
        return prediction, uncertainty

    def predict(self, X, n_iter=50):
        """
        Make a prediction for the next time step.
        
        Args:
            X (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted next vector.
        """
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(1, X.shape[0], X.shape[1])
        y_pred_scaled, uncertainty_scaled = self.predict_with_uncertainty(X_scaled, n_iter=n_iter)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).reshape(-1, X.shape[1])
        uncertainty = self.scaler.inverse_transform(uncertainty_scaled).reshape(-1, X.shape[1])
        return y_pred, uncertainty

    def predict_some(self, data, n_iter=50, steps=1):
        predictionsD = np.array(data)
        predictionsU = np.array([[0,0,0,0]])
        for i in range(steps):
            X_new = predictionsD[-self.n_steps:]
            y_pred, uncertainty = self.predict(X_new, n_iter=n_iter)
            predictionsD = np.append(predictionsD, y_pred, axis=0)
            predictionsU = np.append(predictionsU, uncertainty, axis=0)

        return predictionsD[-steps:], predictionsU


    def expected_error(self, data):
        """
        Calculate the expected error on the validation set.
        
        Args:
            data (np.ndarray): Input data.
        
        Returns:
            float: Expected Mean Absolute Error.
        """
        X_train, y_train, X_val, y_val = self.prepare_dataset(data)
        y_val_pred = self.model.predict(X_val)
        y_val_flat = y_val.reshape(-1, y_val.shape[-1])
        y_val_pred_flat = y_val_pred.reshape(-1, y_val_pred.shape[-1])
        return mean_absolute_error(y_val_flat, y_val_pred_flat)

    def smooth_data_sma(self, data, window_size):
        """
        Applies a Simple Moving Average (SMA) smoothing to the given data over a specified window size.

        Args:
            data (np.ndarray): The input data.
            window_size (int): The size of the window over which to calculate the moving average.

        Returns:
            np.ndarray: The data array after applying SMA.
        """
        smoothed_data = np.copy(data)
        for i in range(data.shape[1]):
            smoothed_col = np.convolve(data[:, i], np.ones(window_size)/window_size, mode='valid')
            smoothed_data[:len(smoothed_col), i] = smoothed_col
        return smoothed_data

    def smooth_data_ema(self, data, alpha):
        """
        Applies an Exponential Moving Average (EMA) smoothing to the given data with a specified smoothing factor `alpha`.

        Args:
            data (np.ndarray): The input data.
            alpha (float): The smoothing factor for the exponential moving average. The value of `alpha` should be between 0 and 1.

        Returns:
            np.ndarray: The data array after applying EMA.
        """
        ema_data = np.zeros_like(data)
        ema_data[0] = data[0]
        for t in range(1, len(data)):
            ema_data[t] = alpha * data[t] + (1 - alpha) * ema_data[t-1]
        return ema_data

