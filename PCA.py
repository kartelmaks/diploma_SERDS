import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import svd
plt.style.use('ggplot')

class PCA:
    def pca(self, n_com, data):

        U, s, Vt = svd(data, full_matrices=0)
        S = np.diag(s)

        T = U.dot(S)[:, :n_com]
        P = Vt.T[:, :n_com]
        return T, P

    def image(self, n_com, data, P, X, x_axes, labels):
        #n_com = 6
        #data = fluor_df
        #U, s, Vt = svd(data, full_matrices=0)
        #S = np.diag(s)
        #P = Vt.T[:, :n_com]
        #X_all = [X_df]
        #labels = ['fluor_df']

        fig_, axes_ = plt.subplots(nrows=2, ncols=n_com, figsize=(100, 20))
        y1 = x_axes.T[0]
        for r in range(P.shape[1]):
            axes_[0, r].set(title="ГК " + str(r + 1))
            axes_[0, r].plot(y1, P.T[r, :])
        k = 0
        for key in labels:
            Q = np.linalg.lstsq(P, X[k].T)[0].T

            for i in range(n_com):
                Q_ = Q[:, i].reshape(58, 141)
                axes_[k + 1, i].imshow(Q_)
                axes_[k + 1, 0].set_ylabel(key)
            k += 1

