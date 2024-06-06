import sys
import json
import numpy as np

def load_json():
	if len(sys.argv)<2:
		return

	with open(sys.argv[1]) as f:
		jContent = np.array(json.load(f))

		bxData = jContent[:,1:-3] #leave only candle data from BINGX packed data
		return bxData
