'''
Get actual trade data as input
'''

import time
import requests
import hmac
from hashlib import sha256

APIURL = "https://open-api.bingx.com"
APIKEY = ""
SECRETKEY = ""

def demo():
    payload = {}
    path = '/openApi/spot/v2/market/kline'
    method = "GET"
    paramsMap = {
    "symbol": "BTC-USDT",
    "interval": "1m",
    "limit": 1000
}
    paramsStr = parseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)

def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
    print("sign=" + signature)
    return signature


def send_request(method, path, urlpa, payload):
    url = "%s%s?%s&signature=%s" % (APIURL, path, urlpa, get_sign(SECRETKEY, urlpa))
    print(url)
    headers = {
        'X-BX-APIKEY': APIKEY,
    }
    response = requests.request(method, url, headers=headers, data=payload)
    return response.text

def parseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    if paramsStr != "": 
     return paramsStr+"&timestamp="+str(int(time.time() * 1000))
    else:
     return paramsStr+"timestamp="+str(int(time.time() * 1000))



import _fixpath
from _plotlib import *

import json
import numpy as np
from vector_prediction import VectorSignalPredictor

# Initialize the predictor
n_steps = 20  # Number of previous steps to use for prediction
predictor = VectorSignalPredictor(n_steps=n_steps, dropout_rate=0.2)

# Generate some synthetic data for demonstration purposes
jdata = json.loads(demo())
data = np.array(jdata['data'])

# Train the model
predictor.fit(data, epochs=20, split_ratio=0.8)

factor = .01
data = predictor.smooth_data_ema(data, factor/(factor+1.0))

# Predict the next step
X_new = data[-n_steps:]  # Take the last n_steps vectors
y_pred, uncertainty = predictor.predict(X_new, 100)


# Print prediction and uncertainty
print(X_new[-5:,:])
print("Predicted next vector:", y_pred)
print("Prediction uncertainty (standard deviation):", uncertainty)

# Calculate and print expected error
expected_error = predictor.expected_error(data)
print("Expected Mean Absolute Error on Validation Set:", expected_error)


plot_vectors(pd.DataFrame(data[:,1:-3]))

