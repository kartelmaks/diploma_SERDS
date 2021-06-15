import pandas as pd
import numpy as np
from SERDS import SERDS
from Plot import Plot

X1 = np.array(pd.read_csv('V784.csv', header=None)[1])
X2 = np.array(pd.read_csv('V784_5.csv', header=None)[1])
X3 = np.array(pd.read_csv('V785.csv', header=None)[1])
X_axes = pd.read_csv('V784.csv', header=None)[0]

serds = SERDS([X1, X2, X3], [0, -6, -12])

raman, fluor = serds.get_reconstr_spectrs()

Plot.plot_all([X1, X2, X3], raman, fluor, X_axes, legend = ['784 нм', '785 нм', '786 нм'])

Plot.plot_spectrs([X1, X2, X3], X_axes)