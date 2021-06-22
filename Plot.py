import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SERDS import serds


class plot(serds):

    def plot_all(spectrs, raman, fluor, x_axis, legend):

        plt.figure(figsize=(12, 7))
        plt.subplot(2, 2, 1)
        plt.plot(x_axis, raman, label='КРС')
        plt.xlabel("$хвильове  число(зсув), см^{-1}.$")
        plt.ylabel("$інтенссивність, у.о.$")
        plt.plot(x_axis, fluor, label='Флуоресц')
        plt.legend(loc=7)
        plt.subplot(2, 2, 2)
        plt.xlabel("$хвильове  число, см^{-1}.$")
        plt.ylabel("$інтенссивність, у.о.$")
        plt.legend(loc=7)

        X_ = pd.DataFrame(spectrs)
        legend = legend

        for i in range(len(X_)):
            Y_axis = np.array(X_.iloc[i, :], dtype=np.float32)
            plt.plot(x_axis, Y_axis, label=legend[i])
        plt.legend(loc=1)
        plt.show()

    def plot_spectrs(arr, x_axis):
        X_ = pd.DataFrame(arr)
        for i in range(len(X_)):
            Y_axis = np.array(X_.iloc[i, :], dtype=np.float32)
            X_axis = np.array(x_axis)
            plt.plot(X_axis, Y_axis)
        plt.show()
