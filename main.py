import pandas as pd
import numpy as np

from SERDS import serds
from Plot import plot
from PCA import pca

X1 = np.array(pd.read_csv('V784.csv', header=None)[1])
X2 = np.array(pd.read_csv('V784_5.csv', header=None)[1])
X3 = np.array(pd.read_csv('V785.csv', header=None)[1])
X_axis = pd.read_csv('V784.csv', header=None)[0]

serds = serds([X1, X2, X3], [0, -6, -12])

raman, fluor = serds.get_reconstr_spectrs()

#plot.plot_all([X1, X2, X3], raman, fluor, X_axis, legend=['784 нм', '785 нм', '786 нм'])
#plot.plot_spectrs([X1, X2, X3], X_axes)

#-----
X_df = pd.read_csv('Export-Vpol-784WL.csv', header=None)
raman_df = pd.read_csv('raman.csv', header=None)
fluor_df = pd.read_csv('fluor.csv', header=None)
x = pd.read_csv('dd.csv', header = None)

T, P = pca.get_pca(6, raman_df)
pca.image(6, P, X_df, x, ['raman'])