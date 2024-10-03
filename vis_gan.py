import sys
sys.path.append( './stylegan2-ada-pytorch')

from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import torch
from vis_basic import ENNinvTool
import pickle

import os




class MinMaxScaler_T(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, feature_range=[0, 1]):
        self.feature_range = feature_range

    def fit(self, tensor):
        """Fit features

        Returns
        -------
        tensor
            A tensor with scaled features using requested preprocessor.
        """
        self.min = torch.tensor(tensor.min(dim=0)[0])
        self.max = torch.tensor(tensor.max(dim=0)[0])
        return self

        
    def fit_transform(self, tensor):
        """Fit and transform features"""
        self.fit(tensor)
        return self.transform(tensor)
    
    def transform(self, tensor):
        """Transform features

        Returns
        -------
        tensor 
            A tensor with scaled features using requested preprocessor.
        """
        with torch.no_grad():
            min = self.min.to(tensor.device)
            max = self.max.to(tensor.device)
            return (tensor - min) / (max - min)
        
    def inverse_transform(self, tensor):
        """Inverse transform features

        Returns
        -------
        tensor 
            A tensor with scaled features using requested preprocessor.
        """
        with torch.no_grad():
            max = self.max.to(tensor.device)
            min = self.min.to(tensor.device)
            return tensor * (max - min) + min
    


class ENNinvTool_G(ENNinvTool):

    def __init__(self, clf=None, Pinv=None, current_z=None, X=None, y=None, X2d=None, GRID=100, padding=0.1, cmap='tab10', show3d=True, device=None, data_shape=None, G_path=None, w_scaler=None) -> None:
        self.w_scaler = w_scaler
        with open(G_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda() 
        super().__init__( clf=clf, Pinv=Pinv, current_z=current_z, X=X, y=y, X2d=X2d, GRID=GRID, padding=padding, cmap=cmap, show3d=show3d, device=device, data_shape=data_shape)
        # name of the window
        
        

     
    # def get_inverse(self, X2d, z=None, GPU=False, batch_size=32):
    #     X2d_scaled = self.scaler2d.inverse_transform(X2d) ##??????
    #     res = self.Pinv.inverse_transform(X2d_scaled, z=z, GPU=True).to(self.device)
    #     w_inv = self.w_scaler.inverse_transform(res)
    #     w_inv = res.unsqueeze(1)
    #     # repeat w_inv to match the shape of w
    #     w_inv = w_inv.repeat(1, self.G.mapping.num_ws, 1)

    #     synth_images = []
    #     for i in range(0, w_inv.shape[0], batch_size):
    #         batch_img = self.G.synthesis(w_inv[i:i+batch_size], noise_mode='const', force_fp32=True)
    #         batch_img = (batch_img + 1) * (255/2)
    #         batch_img = batch_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    #         synth_images.append(batch_img)
    #     synth_images = np.concatenate(synth_images, axis=0)

    #     return synth_images
    
    
    def get_real_data(self, index):
        """
        Overwrite this function when the case is different
        """
        w = self.X[index].to(self.device)
        w = w.unsqueeze(0)
        synth_image = self.gen_img_from_w(w)[0]
        ## flip the image
        synth_image = np.flip(synth_image, axis=0) 
        return synth_image
    
    def update_inverse_window(self, pos, ob=None):
        if pos is None:
            return
        # print(pos)
        if type(pos) == np.ndarray:
            self.mouse_pos = np.array([[pos[0], pos[1]]])
            # print(f'You clicked on image at position: {self.mouse_pos}')
        else:
            viewbox = self.win1.getPlotItem().getViewBox()
            view_coords = viewbox.mapSceneToView(pos)
            # print(f'You clicked on image at position: {view_coords.x()}, {view_coords.y()}')
            self.mouse_pos = np.array([[view_coords.x(), view_coords.y()]])
        # print('mouse pos', self.mouse_pos)
        ## highlight the mouse position on the image
        if ob is None:
            self.mouse_scatter.setData(self.mouse_pos[:,0], self.mouse_pos[:,1], symbol='+', size=20, brush='r')
        else:
            ## plot text 
            self.obtext_list[ob].setPos(self.mouse_pos[0, 0], self.mouse_pos[0, 1])

        cur_z = self.get_z(self.mouse_pos)
        cur_inv = self.get_inverse(self.mouse_pos, cur_z, GPU=True)
        cur_inv = self.gen_img_from_w(cur_inv)     
        cur_inv = cur_inv.reshape(self.data_shape)
        cur_inv = np.flip(cur_inv, axis=0)

        self.inverse_onclick.setImage(cur_inv)
            
        if ob is not None:
            self.ob_list[ob].setImage(cur_inv)
            self.ob_ind = None
            # update ob2d_list
            self.ob2d_list[ob] = self.mouse_pos[0]

    def gen_img_from_w(self, w):
        if type(w) == np.ndarray:
            w = torch.from_numpy(w).to(self.device)
        ### scale w
        w = self.w_scaler.inverse_transform(w)
        w = w.unsqueeze(1)
        w = w.repeat(1, self.G.mapping.num_ws, 1)
        synth_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        synth_images = (synth_images + 1) * (255/2)
        synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return synth_images

    def update_ob_windows(self):
        ob_z = self.get_z(self.ob2d_list)
        ob_nd_lsit = self.get_inverse(self.ob2d_list, z=ob_z, GPU=False)
        ob_nd_lsit = self.gen_img_from_w(ob_nd_lsit)
        for i in range(len(self.ob_list)):
            if sum(self.ob2d_list[i]) != 2: ##
                img = ob_nd_lsit[i].reshape(self.data_shape)
                img = np.flip(img, axis=0)
                self.ob_list[i].setImage(img)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.ctrl_pressed = False
            print('ctrl released')
        elif event.key() == QtCore.Qt.Key_Shift:
            self.select_mode = False
            print('shift released')
        ## == T:
        elif event.key() ==  QtCore.Qt.Key_T:
            print('T released | save x_t, G,(x_t) q_0, G(q_0)')
            ## mkdir 
            os.makedirs(f'./cache/mannual_save/{self.mouse_pos[0][0]:.4f}_{self.mouse_pos[0][1]:.4f}_R_{self.shape_radius}', exist_ok=True)
            path = f'./cache/mannual_save/{self.mouse_pos[0][0]:.4f}_{self.mouse_pos[0][1]:.4f}_R_{self.shape_radius}'
            ## save x_t
            np.save(f'{path}/x_t_{self.data_ind}', self.X[self.data_ind].cpu().numpy())
            ## save G(x_t)
            img = self.gen_img_from_w(self.X[self.data_ind].reshape(1,512).to(self.device))
            img = img.squeeze()
            ## save image
            plt.imsave(f'{path}/G_x_t_{self.data_ind}.png', img)
            ## save q_0
            cur_z = self.get_z(self.mouse_pos)
            q0 = self.get_inverse(self.mouse_pos, cur_z)
            np.save(f'{path}/q0_{self.control_silder.value()/100}', q0)
            ## save G(q_0)
            img_inv = self.gen_img_from_w(q0)
            img_inv = img_inv.squeeze()
            plt.imsave(f'{path}/G(q0)_{self.control_silder.value()/100}.png', img_inv)
            
    
    # def update_2d_map(self):
    #     match self.map_content:
    #         case 3: # distance to the initial surface
    #             if self.show3d:
    #                 map_data = self.compute_2d_map()
                    
    #             map_diff = self.get_distance_to_origin(update=True)
    #             self.map2d.setImage(map_diff)

    #         case 0: # no map content
    #             print('no map content')
    #             # set it to a gray image 
    #             map_data = np.ones((self.GRID, self.GRID, 4)) * 128
    #             self.map2d.setImage(map_data)
                
    #         case 4: # gradient map
    #             print('gradient map')
    #             map_data = self.get_gradient_map()
    #             self.map2d.setImage(map_data)
                

    #         case _: # decision map, distance map with confidence
    #             if self.map_content == 1:
    #                 map_data = self.compute_2d_map(proba=False)
    #             elif self.map_content == 2:
    #                 map_data = self.compute_2d_map(proba=True)
    #             time0 = time.time()
    #             self.map2d.setImage(map_data)
    #             print(f'set map2d time: {time.time() - time0}')

    #     if self.show_scatters or self.update_score:
    #         if self.show_scatter_dist or self.update_score:
    #             ## place holder
    #             ## compute X_reco
    #             X_rocon = self.compute_X_recon()
    #             ## upatge score 
    #             self.update_the_scores(X_recon=X_rocon)
    #             ## compute sizes
    #             sizes = self.process_scatter_size(X_recon=X_rocon)
    #         else:
    #             sizes = self.process_scatter_size()
            
    #         self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.c_scatter*255)
        
    #     # if self.ob2d_list.all():
    #     if not self.show3d: ## TO: isolate this ob update
    #         ob_z = self.get_z(self.ob2d_list)
    #         ob_nd_lsit = self.get_inverse(self.ob2d_list, z=ob_z, GPU=False)
    #         ob_nd_lsit = self.gen_img_from_w(ob_nd_lsit)
    #         for i in range(len(self.ob_list)):
    #             if sum(self.ob2d_list[i]) != 2:
    #                 img = ob_nd_lsit[i].reshape(self.data_shape)*255
    #                 img = np.flip(img, axis=0)
    #                 self.ob_list[i].setImage(img)


if __name__ == '__main__':
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
    w = np.load('cache/AFHQv2/projected_w_all.npz')['w']
    w= w.squeeze()
    y = np.zeros(w.shape[0])
    y[5065 : -4593] = 1
    y[-4593: ] = 2

    w = torch.from_numpy(w).float()
    w_scaler = MinMaxScaler_T()
    w_scaled = w_scaler.fit_transform(w)

    X_train, X_test, y_train, y_test = train_test_split(w_scaled, y, train_size=5000, random_state=42)

    print('w max', X_train.max())
    print('w min', X_train.min())
    clf = NNClassifier(input_dim=X_train.shape[1], n_classes=np.unique(y_train).shape[0], layer_sizes=(1024, 512))
    dataset = torch.utils.data.TensorDataset(X_train.to(clf.device), torch.from_numpy(y_train).long().to(clf.device))
    clf.fit(dataset, epochs=150)
    print("training set acc: ", clf.score(X_train, y_train))

    proj.fit(X_train, epochs=150, early_stop=False)
    X2d = proj.X2d
    # mylabels = proj.Pinv.labels

    ### use this for unsaved model
    app = QtWidgets.QApplication.instance()
    if not app:  # Create new instance if it doesn't exist
        app = QtWidgets.QApplication(sys.argv)
    w = ENNinvTool_G(clf=clf, Pinv=proj.Pinv, X=X_train, X2d=X2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(512,512,3), G_path='../stylegan2-ada-pytorch/stylegan2-afhqv2-512x512.pkl', w_scaler=w_scaler, cmap='tab10')
    # w.showMaximized()
    w.show()
    sys.exit(app.exec())



    #### load saved model in .py 
    ### X2D and classifier are not saved
    ### TODO: save the classifier and X2d
    def load_saved_py(folder='./cache/AFHQv2_py'):
        X2d = p.fit_transform(X_train)  
        Pinv = LCIP()
        Pinv.load_model(folder)
        app = QtWidgets.QApplication.instance()
        if not app:
            app = QtWidgets.QApplication(sys.argv)
        w = ENNinvTool_G(clf=None, Pinv=Pinv, X=X_train, X2d=X2d, y=y_train, GRID=GRID, show3d=True, padding=0.1, data_shape=(512,512,3), G_path='../stylegan2-ada-pytorch/stylegan2-afhqv2-512x512.pkl', w_scaler=w_scaler, cmap='tab10')
        w.show()
        sys.exit(app.exec())

    # load_saved_py()
