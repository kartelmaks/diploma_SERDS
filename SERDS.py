import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import nnls


class SERDS:
    def spline(self, x, y):
        z = np.polyfit(x, y, 10)
        f = np.poly1d(z)

        x_new = np.linspace(0, 1340, 1340)
        y_new = f(x_new)
        return (y_new)

    def extrapolate(self, spectrs):
        sto1 = np.array([[list(range(100))]])
        sto1 = sto1[0, 0, :]
        sto2 = np.array([[list(range(1240, 1340))]])  # винести
        sto2 = sto2[0, 0, :]

        Maximums = list()
        Minimums = list()
        for i in range(3):
            min_arr = np.array(sp.signal.argrelextrema(spectrs[i], np.less))[0]
            L = list()
            for i in range(sto1.shape[0]):
                L.append(sto1[i])
            for i in range(min_arr.shape[0]):
                L.append(min_arr[i])
            for i in range(sto2.shape[0]):
                L.append(sto2[i])

            Minimums.append(L)

        for i in range(3):
            Maximums.append(sp.signal.argrelextrema(spectrs[i], np.greater))

        min_list_0 = spectrs[0][Minimums[0]]
        min_list_1 = spectrs[1][Minimums[1]]
        min_list_2 = spectrs[2][Minimums[2]]

        max_list_0 = spectrs[0][Maximums[0][0]]
        max_list_1 = spectrs[1][Maximums[1][0]]
        max_list_2 = spectrs[2][Maximums[2][0]]

        l_min_1 = spline(Minimums[0], min_list_0)
        l_min_2 = spline(Minimums[1], min_list_1)
        l_min_3 = spline(Minimums[2], min_list_2)

        dr12 = sum(max_list_1) / sum(max_list_0)
        dr13 = sum(max_list_2) / sum(max_list_0)
        dr = [1, dr12, dr13]

        df12 = l_min_2 / l_min_1
        df13 = l_min_3 / l_min_1
        df = [(np.zeros(N) + 1), df12, df13]
        return dr, df

    def H_matrix(self):
        dr, df = extrapolate(spectrs)
        m = 3
        shift = [0, -7, -20]
        ################
        I_arr_ = list()
        E_arr_ = list()

        # E_0 = np.eye(N, N, 0)
        for i in range(m):
            E_ = np.diag(df[i])
            E_arr_.append(E_)

        for r in range(m):
            I_ = np.eye(1340, 1340, shift[r])
            I_arr_.append(I_)

        H1 = I_arr_[0]
        for i in range(m - 1):
            H1 = np.vstack((H1, I_arr_[i + 1] * dr[i + 1]))

        H2 = E_arr_[0]
        for i in range(m - 1):
            H2 = np.vstack((H2, E_arr_[i + 1]))

        H_ = np.hstack((H1, H2))
        ################

        R_ = np.hstack((f[0], f[1]))
        R_ = np.hstack((R_, f[2]))
        ##########################################################
        end, d = nnls(H_, R_)
        Raman.append(end[0:N])
        Fluor.append(end[N:2 * N])