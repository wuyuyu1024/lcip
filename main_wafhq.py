import sys
import os
import numpy as np
import torch
from PySide6 import QtWidgets

from sklearn.model_selection import train_test_split

from umap import UMAP
from sklearn.manifold import TSNE, MDS, Isomap

from classifiers import NNClassifier

from invprojection import Pinv_ilamp, NNinv_torch, RBFinv
from lcip import LCIP
from gui import LCIP_GUI_GAN, MinMaxScaler_T



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
    



GRID = 100

## print current working directory
print('Current working directory: ', os.getcwd())
y = np.load('./datasets/w_afhqv2/labels.npy')
w = np.load('./datasets/w_afhqv2/w_afhqv2.npy')
print(w.shape)

w = torch.from_numpy(w).float()
w_scaler = MinMaxScaler_T()
w_scaled = w_scaler.fit_transform(w)


P_dict ={
    'umap': UMAP(n_components=2, random_state=420),
    'tsne': TSNE(n_components=2, random_state=420, n_jobs=8),
}


Pinv_dict = {
    'ilamp': Pinv_ilamp(k=6),
    'nninv': NNinv_torch(),
    'rbf': RBFinv(),
    'lcip': LCIP(beta=0.01)
}

def train_new_model(P_name='umap', Pinv_name='lcip', clf=None):
    X_train, X_test, y_train, y_test = train_test_split(w_scaled, y, train_size=5000, random_state=42)

    P = P_dict[P_name]
    Pinv = Pinv_dict[Pinv_name]
    proj = Simple_P_wrapper(P, Pinv)

    ### train a classifier (for making decision maps)
    if clf:
        print('Training the classifier')
        clf = NNClassifier(input_dim=X_train.shape[1], n_classes=np.unique(y_train).shape[0], layer_sizes=(512, 256, 128))
        dataset = torch.utils.data.TensorDataset(X_train.to(clf.device), torch.from_numpy(y_train).long().to(clf.device))
        clf.fit(dataset, epochs=100)

    print(f'Fitting the projection [{P_name}] and the inverse projection [{Pinv_name}]')
    proj.fit(X_train, epochs=120, early_stop=False) ## train the projection and the inverse
    X2d = proj.X2d

    app = QtWidgets.QApplication.instance()
    if not app:  # Create new instance if it doesn't exist
        app = QtWidgets.QApplication(sys.argv)

    ## set clf to the trained classifier to use it for decision maps

    w = LCIP_GUI_GAN(clf=clf, Pinv=proj.Pinv, X=X_train, X2d=X2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(512,512,3), G_path='models/stylegan2-afhqv2-512x512.pkl', w_scaler=w_scaler, cmap='tab10')
    # w.showMaximized()
    w.resize(1700, 860)
    w.show()
    sys.exit(app.exec())


## load saved model
def load_saved(folder, clf=None):
    Pinv = LCIP()
    data_dict = np.load(folder + '/data_dict.npz')
    
    X_train = data_dict['X_train']
    X_2d = data_dict['X2d_unscaled']
    y_train = data_dict['y_train']
    X_train = torch.from_numpy(X_train).float()

    Pinv.load_model(folder, input_dim=X_train.shape[1])
        
    if clf:
        ### train a classifier (for making decision maps)
        print('Training the classifier')
        clf = NNClassifier(input_dim=X_train.shape[1], n_classes=np.unique(y_train).shape[0], layer_sizes=(512, 256, 128))
        dataset = torch.utils.data.TensorDataset(X_train.to(clf.device), torch.from_numpy(y_train).long().to(clf.device))
        clf.fit(dataset, epochs=100)

    
    app = QtWidgets.QApplication.instance()
    if not app:  
        app = QtWidgets.QApplication(sys.argv)
    print('shape X:' , X_train.shape)

    window = LCIP_GUI_GAN(clf=clf, Pinv=Pinv, X=X_train, X2d=X_2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(512,512,3), G_path='./models/stylegan2-afhqv2-512x512.pkl', w_scaler=w_scaler, cmap='tab10')

       
    window.resize(1700, 860)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_paper', action='store_true', help='load the saved model from the paper')
    parser.add_argument('-p', '--projection', type=str, help='Choose the projection method. The default is umap', choices=['umap', 'tsne' ], default='tsne')
    parser.add_argument('-i', '--pinv', type=str, help='Choose the inverse projection method. The default is lcip', choices=['ilamp', 'nninv', 'rbf', 'lcip'], default='lcip')
    parser.add_argument('-c', '--clf', action='store_true', help='Train a classifier for decision maps')
    args = parser.parse_args()
    print(args.load_paper)
    if args.load_paper:
        load_saved('./models/wAFHQv2_paper', clf=args.clf)
    else:
        train_new_model(args.projection, args.pinv, clf=args.clf)


# load_saved('./models/wAFHQv2_paper')