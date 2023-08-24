import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

pd.set_option('display.max_columns', None)
# intData = pd.read_csv("dateInterogate.csv")
# print(intData)
horizon = 30
model = LSTM(h=horizon)