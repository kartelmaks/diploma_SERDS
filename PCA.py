import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.signal import savgol_filter
from plotly.graph_objs import *
import plotly.offline as offline

from sklearn.decomposition import PCA as PCA
from scipy.linalg import svd

plt.style.use('ggplot')

class PCA:
    def pca(self, n_com, data):

        U, s, Vt = svd(data, full_matrices=0)
        S = np.diag(s)

        T = U.dot(S)[:, :n_com]
        P = Vt.T[:, :n_com]


