import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from SOM import SOM
from sklearn.datasets import load_digits


X, y = digits = load_digits(return_X_y=True)


som = SOM(10, 10)
som.fit(X, y)
som.plot()









