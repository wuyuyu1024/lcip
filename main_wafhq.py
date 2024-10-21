import sys
# make blob dataimport sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
# from ENNinv import  PPinv_wrapper, DisentangledNNinv
from lcip import LCIP
from classifiers import NNClassifier
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib import cm
# from NNinv import NNinv_torch, KNN_Pinv

from sklearn.manifold import TSNE, MDS, Isomap
# import PCA
from sklearn.decomposition import PCA
import torch
# from lamp import Pinv_ilamp
# swiss roll
from sklearn.datasets import make_swiss_roll
# from rbf_inv import RBFinv
import torchvision
import joblib
import numpy as np
from gui import LCIP_GUI_GAN, MinMaxScaler_T
from PySide6 import QtWidgets


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
        try:
            pass
            self.Pinv.load_model('./cache/AFHQv2_rbf_placeholder')
        except:
            self.Pinv.fit(self.X2d, x, **kwargs)
        return self
    

# p = UMAP(n_components=2, random_state=420) #  min_dist=0.9, , n_neighbors=50
p = TSNE(n_components=2, random_state=420, n_jobs=8)
# p = PCA(n_components=2)
# p = MDS(n_components=2, random_state=420, n_jobs=8)
# p = Isomap(n_components=2, n_jobs=8)

# p = ShaRP(original_dim=32*32*3, n_classes=10, 
#                 variational_layer="diagonal_normal",
#                 variational_layer_kwargs=dict(kl_weight=0.1),
#                 bottleneck_activation="linear",
#                 # bottleneck_l1=0.0,
#                 # bottleneck_l2=0.5,
#         )


# proj = Simple_P_wrapper(p, NNinv_torch())#[256, 256, 256, 256]
# proj = Simple_P_wrapper(p, ENNinv(z_dim=2))
# proj = Simple_P_wrapper(p, DisentangledNNinv(z_dim=64, mini_epochs=5, beta=0.01, z_neighbor=10, z_finder_method='rbf', layers_e=[512, 512, 512, 512], layers_d=[512, 512, 512, 512], weight_y=0, use_BN=False)) # 3d
proj = Simple_P_wrapper(p, LCIP(z_dim=16, mini_epochs=5, beta=0.01, z_neighbor=10, z_finder_method='rbf', layers_e=None, layers_d=None, weight_y=0, use_BN=False)) # AFHQv2 

# proj = Simple_P_wrapper(p, KNN_Pinv()) # coil20
# proj = Simple_P_wrapper(p, Pinv_ilamp(k=6))
# proj = Simple_P_wrapper(p, RBFinv(function_type='linear', scipy=True))
# proj = Simple_P_wrapper(p, RBFinv())
# proj = Simple_P_wrapper(p, RBFinv(function_type='thin_plate_spline', scipy=True))
GRID = 100

# w = np.load('../batchProject_styleGAN2/out_batch/projected_w_all.npz')['w']
w = np.load('datasets/w_afhqv2/w_afhqv2.npy')
w = w.squeeze()
print(w.shape)
y = np.load('datasets/w_afhqv2/labels.npy')

w = torch.from_numpy(w).float()
w_scaler = MinMaxScaler_T()
w_scaled = w_scaler.fit_transform(w)

X_train, X_test, y_train, y_test = train_test_split(w_scaled, y, train_size=5000, random_state=42)
# X_train, y_train = w_scaled, y

print('w max', X_train.max())
print('w min', X_train.min())
clf = NNClassifier(input_dim=X_train.shape[1], n_classes=np.unique(y_train).shape[0], layer_sizes=(512, 256, 128))
dataset = torch.utils.data.TensorDataset(X_train.to(clf.device), torch.from_numpy(y_train).long().to(clf.device))
# clf.fit(dataset, epochs=150)
print("training set acc: ", clf.score(X_train, y_train))

proj.fit(X_train, epochs=150, early_stop=False)
X2d = proj.X2d
# mylabels = proj.Pinv.labels

### use this for unsaved model
app = QtWidgets.QApplication.instance()
if not app:  # Create new instance if it doesn't exist
    app = QtWidgets.QApplication(sys.argv)
w = LCIP_GUI_GAN(clf=None, Pinv=proj.Pinv, X=X_train, X2d=X2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(512,512,3), G_path='models/stylegan2-afhqv2-512x512.pkl', w_scaler=w_scaler, cmap='tab10')
# w.showMaximized()
w.show()
sys.exit(app.exec())
