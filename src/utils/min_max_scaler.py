import pandas as pd
import numpy as np
class MinMaxScaler:

    def __init__(self, dataframe: pd.DataFrame):
        self._Y_values = dataframe['y'].to_numpy()
        self._min = np.min(self._Y_values)
        self._max= np.max(self._Y_values)

    def scale(self) -> np.ndarray:
        return (self._Y_values - self._min)/(self._max - self._min)