import pandas as pd
import numpy as np
from SERDS import SERDS

X1 = np.array(pd.read_csv('V784.csv', header = None)[1])
X2 = np.array(pd.read_csv('V784_5.csv', header = None)[1])
X3 = np.array(pd.read_csv('V785.csv', header = None)[1])

serds = SERDS([X1, X2, X3], [0, -6, -12])

raman, fluor = serds.get_reconstr_spectrs()


X_shift__ = pd.read_csv('V784.csv', header = None)[0]
import matplotlib.pyplot as plt
X_axis = (10**7)/X_shift__
plt.figure(figsize=(12, 7))
# Вывод графиков
plt.subplot(2, 2, 1)
plt.plot((10**7/784) - X_axis, raman, label = 'КРС')
plt.xlabel("$хвильове  число(зсув), см^{-1}.$")
plt.ylabel("$інтенссивність, у.о.$")
plt.plot((10**7/784) - X_axis, fluor, label = 'Флуоресц')
plt.legend(loc = 7)
plt.subplot(2, 2, 2)
plt.xlabel("$хвильове  число, см^{-1}.$")
plt.ylabel("$інтенссивність, у.о.$")
plt.legend(loc = 7)
X_ = pd.DataFrame([X1, X2, X3])
legend = ['784 нм', '785 нм', '786 нм']
for i in range(len(X_)):
    Y_axis = np.array(X_.iloc[i,:], dtype = np.float32)
    #X_axis = np.array(range(N))
    X_axis = X_axis
    #X_axis = X_shift
    plt.plot(X_axis, Y_axis, label = legend[i])
plt.legend(loc = 1)
plt.show()