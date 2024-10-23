import sys
import os
import torch
import numpy as np
from PySide6 import QtWidgets

from sklearn.datasets import make_blobs, make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE, MDS, Isomap
from umap import UMAP
from sklearn.model_selection import train_test_split

from classifiers import NNClassifier

# from NNinv import NNinv_torch, KNN_Pinv
# from lamp import Pinv_ilamp
# from rbf_inv import RBFinv
from lcip import  LCIP
from gui import LCIP_GUI_Basic



class Simple_P_wrapper:
    def __init__(self, P, Pinv):
        self.P = P
        self.Pinv = Pinv
    def __call__(self, x):
        return self.P(x)
    def transform(self, x):
        return self.P.transform(x)
    # with keywrod argumentscon
    def inverse_transform(self, x, **kwargs):
        return self.Pinv.transform(x, **kwargs)

    def fit(self, x, x2d=None, **kwargs):
        if x2d is None:
            self.X2d = self.P.fit_transform(x).astype('float32')
        else:
            self.X2d = x2d
        self.Pinv.fit(self.X2d, x, **kwargs)
        return self
    


# p = UMAP(n_components=2, random_state=420) #  min_dist=0.9, , n_neighbors=50
p = TSNE(n_components=2, random_state=420, n_jobs=8)
# p = PCA(n_components=2)
# p = MDS(n_components=2, random_state=420, n_jobs=8)
# p = Isomap(n_components=2, n_jobs=8)


proj = Simple_P_wrapper(p, LCIP(z_dim=16, mini_epochs=5, beta=0.1, z_neighbor=10, z_finder_method='rbf', layers_e=None, layers_d=None)) # MNIST
# proj = Simple_P_wrapper(p, NNinv_torch())#[256, 256, 256, 256]
# proj = Simple_P_wrapper(p, KNN_Pinv()) # coil20
# proj = Simple_P_wrapper(p, Pinv_ilamp(k=6))
# proj = Simple_P_wrapper(p, RBFinv())
GRID = 100


# blob
# X_train, y_train = make_blobs(n_samples=500, centers=6, n_features=3, random_state=0, cluster_std=1.2)

# MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = np.array(X)
y = np.array(y)
X = X.astype('float32') / 255.
y = y.astype('int')
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3000, test_size=2000, random_state=420)

# ## fashion MNIST
# X, y = fetch_openml('fashion-mnist', version=1, return_X_y=True)
# X = np.array(X)
# y = np.array(y)
# X = X.astype('float32') / 255.
# y = y.astype('int')
# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3000, test_size=2000, random_state=420)


####################

### dont forget to scale the data !
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

## train a classifier (can be used for making decision maps)
clf = NNClassifier(input_dim=X_train.shape[1], n_classes=np.unique(y_train).shape[0], layer_sizes=(200, 50))
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(clf.device), torch.from_numpy(y_train).long().to(clf.device))
clf.fit(dataset, epochs=100)
print(clf.score(X_train, y_train))

## train the projection and the inverse
proj.fit(X_train, epochs=100, early_stop=False)
X2d = proj.X2d


# Check if QApplication already exists
app = QtWidgets.QApplication.instance()
if not app:  # Create new instance if it doesn't exist
    app = QtWidgets.QApplication(sys.argv)
w = LCIP_GUI_Basic(clf=clf, Pinv=proj.Pinv, X=X_train, X2d=X2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(28,28,1))
# w.showMaximized()
w.resize(1700, 860)
w.show()
sys.exit(app.exec())