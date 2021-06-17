import numpy as np
from scipy.optimize import nnls
from scipy.signal import argrelextrema
from Plot import Plot

class SERDS:

    def __init__(self, spectrs, shift):
        self.shift = shift
        self.spectrs = spectrs
        self.N = len(spectrs[0])

    def get_reconstr_spectrs(self):
        R = np.hstack((self.spectrs[0], self.spectrs[1]))
        R = np.hstack((R, self.spectrs[2]))

        H = self.H_matrix()
        end, d = nnls(H, R)

        raman = (end[0:self.N])
        fluor = (end[self.N: 2 * self.N])

        return raman, fluor

    def H_matrix(self):

        dr, df = self.intens_coeff(self.spectrs)
        m = len(self.spectrs)

        I_arr_ = list()
        E_arr_ = list()

        for i in range(m):
            E_ = np.diag(df[i])
            E_arr_.append(E_)

        for r in range(m):
            I_ = np.eye(1340, 1340, self.shift[r])
            I_arr_.append(I_)

        H1 = I_arr_[0]
        for i in range(m - 1):
            H1 = np.vstack((H1, I_arr_[i + 1] * dr[i + 1]))

        H2 = E_arr_[0]
        for i in range(m - 1):
            H2 = np.vstack((H2, E_arr_[i + 1]))

        H_ = np.hstack((H1, H2))
        return H_

    def intens_coeff(self, spectrs):

        sto1 = np.array([[list(range(100))]])
        sto1 = sto1[0, 0, :]
        sto2 = np.array([[list(range(self.N - 100, self.N))]])
        sto2 = sto2[0, 0, :]

        Maximums = list()
        Minimums = list()
        for j in range(3):
            min_arr = np.array(argrelextrema(spectrs[j], np.less))[0]
            L = list()
            for i in range(sto1.shape[0]):
                L.append(sto1[i])
            for i in range(min_arr.shape[0]):
                L.append(min_arr[i])
            for i in range(sto2.shape[0]):
                L.append(sto2[i])

            Minimums.append(L)

        for i in range(3):
            Maximums.append(argrelextrema(spectrs[i], np.greater))

        list_min = [0, 0, 0]
        list_min[0] = spectrs[0][Minimums[0]]
        list_min[1] = spectrs[1][Minimums[1]]
        list_min[2] = spectrs[2][Minimums[2]]

        list_max = [0, 0, 0]
        list_max[0] = spectrs[0][Maximums[0][0]]
        list_max[1] = spectrs[1][Maximums[1][0]]
        list_max[2] = spectrs[2][Maximums[2][0]]

        spline_min = [0, 0, 0]
        spline_min[0] = self.spline(Minimums[0], list_min[0])
        spline_min[1] = self.spline(Minimums[1], list_min[1])
        spline_min[2] = self.spline(Minimums[2], list_min[2])

        dr12 = sum(list_max[1]) / sum(list_max[0])
        dr13 = sum(list_max[2]) / sum(list_max[0])
        dr = [1, dr12, dr13]

        df12 = spline_min[1] / spline_min[0]
        df13 = spline_min[2] / spline_min[0]
        df = [(np.zeros(self.N) + 1), df12, df13]
        return dr, df

    def spline(self, x, y):
        z = np.polyfit(x, y, 10)
        f = np.poly1d(z)

        x_new = np.linspace(0, self.N, self.N)
        y_new = f(x_new)
        return y_new
