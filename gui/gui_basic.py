from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QWidget,QPushButton,QHBoxLayout,QVBoxLayout, QFileDialog
from PySide6.QtCore import Signal
import pyqtgraph as pg
import numpy as np
# import pandas as pd
import matplotlib.cm as cm
from matplotlib import colormaps
# import vispy
from vispy import scene, app
from vispy.scene import visuals
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
# import gaussina process
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import time

import torch
import os

pg.setConfigOption('imageAxisOrder', 'row-major') 


class CustomPlotWidget(pg.PlotWidget):
    # Define a custom signal
    customMousePressed = Signal(object)
    customMouseReleased = Signal(object)


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Emit the custom signal when the mouse is pressed
        self.customMousePressed.emit(event)
        # print("Mouse Pressed in Custom Widget")


    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # self.parent_custom.mouse_pressed = False
        # Emit the custom signal when the mouse is pressed
        self.customMouseReleased.emit(event)
        # print("Mouse Released in Custom Widget")


class NonInteractiveScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super(NonInteractiveScatterPlotItem, self).__init__(*args, **kwargs)

    def mouseClickEvent(self, event):
        pass  # Do nothing

    def mouseMoveEvent(self, event):
        pass  # Do nothing


class LCIP_GUI_Basic(QWidget):

    def __init__(self, Pinv, X, X2d, y=None, data_shape=None, current_z=None, clf=None, GRID=100, padding=0.1, cmap='tab10', show3d=True, device=None) -> None:
        """
        
        Pinv: the inverse projection method, with `inverse_transform` method. Use LCIP for the dynamic controllable inverse projection; otherwise, a static inverse projection will be shown.
        X: the data samples. shape: (n_samples, n_features) 
        X2d: the 2d representation of the data samples (accuired by t-SNE, UMAP, etc.)
        y: the labels of the data samples
        current_z: the initial z values for the LCIP method (if None, it will be initialized by the inverse projection method). Only needed when Pinv is LCIP.
        clf: the classifier for the decision map (with sklearn API) | Optional | Default: None
        GRID: the grid size for the decision map | Optional | Default: 100
        padding: the padding for the decision map | Optional | Default: 0.1
        cmap: the colormap for the decision map | Optional | Default: 'tab10'
        show3d: whether to show the 3d decision map | Optional | Default: True | Note: it will be set to False if the data samples have more than 3 dimensions
        device: the device for the computation | Optional | Default: None | Note: it will be set to 'cuda' if the GPU is available, otherwise 'cpu'
        data_shape: the shape of the output image. Note: if G is not None, it should be the same as G's output shape     
        """

        super().__init__()
        # name of the window
        self.setWindowTitle('Interactive Decision Map')
        size = 18
        self.setStyleSheet(f'font-size: {size}px;') 

        ## set aspect ratio to 16:9
        self.setBaseSize(1600, 900)
        
        self.GRID = GRID

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #### data stuff, could be moved to a dict
        self.X = X
        if data_shape is None:
            self.data_shape = X.shape[1:]
        else:
            self.data_shape = data_shape

        # self.side_size = int(np.sqrt(X.shape[1])) if int(np.sqrt(X.shape[1]))**2 == X.shape[1] else 3
        self.neighbor_finder = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X)
        self.y = y
        self.scaler2d = MinMaxScaler()
        self.X2d = self.scaler2d.fit_transform(X2d)   
        self.cmap = colormaps[cmap]
        # self.c_scatter = self.cmap(y/np.max(y))  
        if cmap == 'tab10':
            y = y / 9
        else:
            y = y / y.max()
        self.ponit_color_mode = 'label'
        self.c_scatter = self.cmap(y)  
        self.max_dist = None   
        self.show_scatter_dist = False
        self.acc_score = None
        self.scatter_size_scaler = None

        ### resetable variables
        self.clf = clf
        self.Pinv = Pinv
        self.show3d = show3d
        if X.shape[1] > 4:
            self.show3d = False

        self.padding = padding
        xx, yy = np.meshgrid(np.linspace(0-self.padding, 1+self.padding, self.GRID), np.linspace(0-self.padding, 1+self.padding, self.GRID))
        self.XY_grid = np.c_[xx.ravel(), yy.ravel()] # shape (GRID*GRID, 2)
        # print(self.XY_grid)
        

        if current_z is None:
            # self.current_z = np.zeros((GRID*GRID, PPinv.Pinv.z_dim))
            self.current_z = self.init_z()
        else:
            self.current_z = current_z

        # back up the current_z
        self.initial_z = self.current_z.copy()  
        self.initial_Iinv = self.get_inverse(self.XY_grid, self.initial_z, GPU=True)   # TODO: check if this is necessary      
        print('initial Iinv shape:', self.initial_Iinv.shape)
        print('initial Iinv device:', self.initial_Iinv.device)

        ## temp variables
        self.mouse_pos = None
        self.data_ind = None
        
        ## slider variables
        self.scatter_size_factor = 20
        self.shape_factor = 0.5
        self.shape_radius = 0.1
        ## add z values


        self.win1, self.map2d, self.scatter2d = self.init_widget_map()
        if self.show3d:
            self.win2, self.surface3d = self.init_widget_3d()
        else:
            ob_col, self.ob_list = self.init_ob_wiget()
            
            self.ob2d_list = np.ones((len(self.ob_list), 2))
            
        self.win3, self.inverse_onclick, self.real_onclick = self.init_widget_side()
        
        # H layout
        hlayout = QtWidgets.QHBoxLayout()
        if self.show3d:
            hlayout.addWidget(self.win2.native, 2)
            hlayout.addWidget(self.win1, 2)
            hlayout.addLayout(self.win3, 1)
        else:
            hlayout.addWidget(self.win1, 4)
            hlayout.addLayout(ob_col, 1)
            hlayout.addLayout(self.win3, 3)
        # force the first one to be square
    
        self.setLayout(hlayout)
        self.z_change_stack = []
        self.mouse_pressed = False
        self.cur_init_z = 0
        self.ctrl_pressed = False
        self.select_mode = False
        self.ob_ind = None
        self.cache_time = 0


    def init_z(self):
        current_z = self.Pinv.find_z(self.XY_grid)
        return current_z
            
    def process_scatter_size(self, X_recon=None):
        if not self.show_scatter_dist:
            return np.ones(self.X2d.shape[0]) * self.scatter_size_factor * 0.5
        
        self.distances = self.get_distance(X_recon=X_recon, GPU=False).astype(np.float16)
        time0 = time.time()
        if self.scatter_size_scaler is None:
            self.scatter_size_scaler = MinMaxScaler()
            raw_data_sizes = self.scatter_size_scaler.fit_transform(self.distances.reshape(-1,1) ) + 0.1
        else:
            raw_data_sizes = self.scatter_size_scaler.transform(self.distances.reshape(-1,1)) + 0.1
    
        
        self.raw_data_sizes = raw_data_sizes.reshape(-1) + 0.1
        sizes = self.scatter_size_factor *  self.raw_data_sizes
        print(f'process_scatter_size time: {time.time() - time0}')
        print(type(sizes))
        return sizes


    def compute_2d_map(self, proba=False):
        time0 = time.time()
        if not hasattr(self, 'cur_Iinv') or not np.array_equal(self.old_XY_grid, self.XY_grid) or not np.array_equal(self.old_current_z, self.current_z):
            self.cur_Iinv = self.get_inverse(self.XY_grid, self.current_z, GPU=True)
            self.old_XY_grid = self.XY_grid.copy()
            self.old_current_z = self.current_z.copy()
        print(f'get_inverse time: {time.time() - time0}')

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_map_data = executor.submit(self.get_prob_map, self.cur_Iinv, proba=proba)
            map_data = future_map_data.result()

        self.map_color = map_data
        return map_data
    
    def get_inverse(self, X2d, z=None, GPU=False):
        X2d_scaled = self.scaler2d.inverse_transform(X2d) ##??????
        res = self.Pinv.inverse_transform(X2d_scaled, z=z, GPU=GPU)
        if type(res) == np.ndarray and GPU:
            res = torch.tensor(res, dtype=torch.float32, device=self.device)
        return res
    
    def get_prob_map(self, inversed_data, proba=True, epsilo=0.8):
        if self.clf is None:
            return np.ones((self.GRID, self.GRID, 4)) * 128
        ########################################
        time0 = time.time()
        probs = self.clf.predict_proba(inversed_data)
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        ###################################

        labels_normlized = labels#/self.clf.classes_.max()
        map = self.cmap(labels_normlized)
        if proba:
            map[:, 3] = alpha 
        map[:, 3] *= epsilo  # plus a float to control the transparency
        map =  map.reshape(self.GRID, self.GRID, 4)
        # map = np.ones((self.GRID, self.GRID, 4))
        print(f'get_prob_map time: {time.time() - time0}')
        return map
    
    def get_gradient_map(self, grid=None):
        """
        get the gradient map for the inverse projection method

        projecters: the inverse projection method. It should have a inverse_transform method that can map the 2d points back to the original space
        x2d: the 2d points. 
        grid: the grid size for the gradient map.
        """
        # make grid
        # x2d = projecters.transform(x)
        if grid is None:
            grid = self.GRID
        
        x_max, x_min = 1 + self.padding, 0 - self.padding
        y_max, y_min = 1 + self.padding, 0 - self.padding
        pixel_width = (x_max - x_min) / grid
        pixel_height = (y_max - y_min) / grid
        # pixel_width =  1/grid
        # pixel_height = 1/grid

        grid_pad = grid + 2 

        xx, yy = np.meshgrid(np.linspace(x_min-pixel_width, x_max+pixel_width, grid_pad), np.linspace(y_min-pixel_height, y_max+pixel_height, grid_pad)) # make it 100*100 to reduce the computation
        xy = np.c_[xx.ravel(), yy.ravel()]
        z_of_XY = self.get_z(xy).astype(np.float16)
        # get the gradient
        ndgrid_padding = self.get_inverse(xy, z=z_of_XY, GPU=True)
        # print(ndgrid_padding.shape)
        
        # ndgrid_rec = ndgrid_rec
        ndgrid_padding = ndgrid_padding.reshape(grid_pad, grid_pad, -1)
        ## remove the padding for gradient map. 
        ## This is the inverse porjection for all the pixels. It can be cached for downstream use, such as decision boundary map
        self.cur_Iinv = ndgrid_padding[1:-1, 1:-1, :].reshape(-1, ndgrid_padding.shape[2])
        print('cur_Iinv shape:', self.cur_Iinv.shape)

        Dx = ndgrid_padding[2:, 1:-1] - ndgrid_padding[:-2, 1:-1]
        Dy = ndgrid_padding[1:-1, 2:] - ndgrid_padding[1:-1, :-2]

        ### original implementation
        # Dx = Dx / (2 * pixel_width)
        # Dy = Dy / (2 * pixel_height)
        ## just assume the pixel width and height are both 1
        Dx = Dx / 2
        Dy = Dy / 2

        # get the gradient norm
        # D = np.sqrt(np.sum(Dx**2, axis=2) + np.sum(Dy**2, axis=2))
        ## torch 
        with torch.no_grad():
            D = torch.sqrt(torch.sum(Dx**2, axis=2) + torch.sum(Dy**2, axis=2))
        ## This is the gradient map according to the equations in UnProjeciton paper 
        # D = D.reshape(-1)

        ## not necessary to normalize the gradient map to [0,1] here
        norm_D = (D / torch.max(D)).cpu().numpy()

        cmap = cm.jet
        D_color = cmap(norm_D)
        self.map_color = D_color
        ## return the gradient map and the inverse projection for all the pixels
        return  D_color
    
    def get_distance_to_neighbor(self):
        self.cur_Iinv = self.get_inverse(self.XY_grid, self.current_z, GPU=True)
        dist, _ = self.neighbor_finder.kneighbors(self.cur_Iinv.cpu().numpy())
        dist = dist.reshape(self.GRID, self.GRID)
        cmap = cm.viridis
        dist_color = cmap(dist/np.max(dist))
        return dist_color


    def compute_X_recon(self):
        time0 = time.time()
        z_of_XY = self.get_z(self.X2d).astype(np.float16)
        reconstructed_GPU = self.get_inverse(self.X2d, z=z_of_XY, GPU=True)
        if type(reconstructed_GPU) == np.ndarray:
            reconstructed_GPU = torch.tensor(reconstructed_GPU, dtype=torch.float16, device=self.device)
        else:
            reconstructed_GPU = reconstructed_GPU.type(torch.float16)
        print(f'compute_X_recon time: {time.time() - time0}')
        return reconstructed_GPU
    
    def update_the_scores(self, X_recon=None):
        time0 = time.time()
        if X_recon is None:
            X_recon = self.compute_X_recon()
        if self.clf is not None:
            # compute the ACC_M score
            y_pred_map = self.clf.predict(X_recon.type(torch.float32))
            acc_score = accuracy_score(self.y, y_pred_map)
            self.acc_score = acc_score
            self.score_label_accm.setText(f'Map Accuracy: {self.acc_score:.4f}')
            # compute the consistency score
            y_pred = self.clf.predict(self.X)
            self.cons_score = accuracy_score(y_pred, y_pred_map)
            
            self.score_label_cons.setText(f'Consistency: {self.cons_score:.4f}')
        print(f'computing score, time: {time.time() - time0}')
    
    def get_distance(self, X_recon=None, GPU=False):   ### this is the most expensive part right now 0.16s for Fashion
        """
        get the distance of data ponits from the ground truth to the current map
        """
        ## get the distance from the current point to the grid points
        ## and compute the ACC_M score 
        ##############GPU version
        time0 = time.time()
        if X_recon is None:
            X_recon = self.compute_X_recon()
   
        # Optimized distance calculation using broadcasting, avoiding explicit loops
        # This assumes Euclidean distance for optimization purposes, can be adjusted if needed
        X = torch.tensor(self.X, dtype=torch.float16, device=self.device)
        X_recon = X_recon.type(torch.float16)

        X_squared = torch.sum(X ** 2, dim=1, keepdim=True)
        recon_squared = torch.sum(X_recon ** 2, dim=1).unsqueeze(1)
        distances = torch.sqrt(X_squared + recon_squared - 2 * torch.matmul(X, X_recon.mT))

        # Extract diagonal for point-wise distances
        distances = distances.diag()
        print('time for distance calculation GPU version:', time.time() - time0)

        # release GPU memory
        del X_recon
        # del distances_gpu
        torch.cuda.empty_cache()
        if GPU:
            return distances
        else:
            return distances.cpu().numpy()


    def get_z(self, X2d):
        #TODO: re fit only when needed
        if X2d is None:
            return
        self.fnn_p2d = KNeighborsRegressor(n_neighbors=4, weights='distance', n_jobs=-1).fit(self.XY_grid, self.current_z)
        z = self.fnn_p2d.predict(X2d.reshape(-1, 2))   
        return z
    

    def encode(self, X):
        return self.Pinv.encode(X)
    

    def gaussian_filter(self, X2d, sigma=0.1):
        """
        X2d now is a 1d array of 2d points
        multiple points situation is not considered yet
        """
        ## two D gaussian filter
        ## XY is the grid points
        ## X2d is the center of the gaussian
        ## sigma is the standard deviation
        ## return a 2d array of gaussian filter
        x = self.XY_grid[:,0]
        y = self.XY_grid[:,1]
        x0 = X2d[0]
        y0 = X2d[1]

        gaussian = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))
        gaussian =  gaussian.reshape(self.GRID, self.GRID, -1)
        # stack filter to the same shape as self.current_z
        filter = np.repeat(gaussian, self.current_z.shape[1], axis=2)
        return filter
        
    
    def update_z(self, location=None):
        if location is None:
            delta_z = self.encode(self.X[self.data_ind]) - self.get_z(self.X2d[self.data_ind])
            filter = self.gaussian_filter(self.X2d[self.data_ind], sigma=self.shape_radius)
        else:
            delta_z = self.encode(self.X[self.data_ind]) - self.get_z(location)
            filter = self.gaussian_filter(location, sigma=self.shape_radius)
        # print(f'filter shape: {filter.shape}')
        
        delta = delta_z * filter
        self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor


    def init_widget_map(self):

        # map_data = self.compute_2d_map()
        map_data = self.compute_2d_map()
        sizes = self.process_scatter_size() ## 
        # print(map_data)
        scatter2d = pg.ScatterPlotItem(self.X2d [:,0], self.X2d [:,1], symbol='o', size=sizes, pen=pg.mkPen(color='k', width=0.5), brush=self.c_scatter*255, )
        map2d = pg.ImageItem(map_data)
        # Scale the image to [0, 1]
        map2d.setRect(QtCore.QRectF(0-self.padding, 0-self.padding, 1+2*self.padding, 1+2*self.padding))

        ###Create a plot and add scatter and image items to it
        # plot = pg.plot()
        plot = CustomPlotWidget()
        plot.addItem(map2d)
        plot.addItem(scatter2d) 
        
        ## #set background color
        plot.setBackground('w')
        ## hide axis
        plot.hideAxis('bottom')
        plot.hideAxis('left')

        ### connect signals
        scatter2d.sigClicked.connect(self.scatter_clicked)
        # # Connect mouse click event to the callback function for the image
        # plot.scene().sigMouseClicked.connect(self.image_clicked)
        plot.customMousePressed.connect(self.image_clicked)
        plot.scene().sigMouseMoved.connect(self.image_moved)
        # plot.sigMouseReleased.connect(self.image_released)
        plot.customMouseReleased.connect(self.image_released)
        
        plot.setLimits(xMin=0-self.padding, xMax=1+self.padding, yMin=0-self.padding, yMax=1+self.padding)

        self.mouse_scatter = NonInteractiveScatterPlotItem()
        self.selected_2dPonit = NonInteractiveScatterPlotItem()
        plot.addItem(self.mouse_scatter)
        plot.addItem(self.selected_2dPonit)

        self.radius_circle = plot.plot()

        return plot, map2d, scatter2d
    
    def plot_circle(self, center=[0.5, 0.5], radius=0.1, color=None):
    # Generate theta values
        theta = np.linspace(0, 2 * np.pi, 1000)

        # Compute x and y coordinates using the parametric equations for a circle
        x1 = radius * np.cos(theta)
        y1 = radius * np.sin(theta)
        x1 += center[0]
        y1 += center[1]

        x2 = radius * np.cos(theta) * 2
        y2 = radius * np.sin(theta) * 2
        x2 += center[0]
        y2 += center[1]

        nan_array = np.array([np.nan])
        x = np.concatenate((x1, nan_array, x2))
        y = np.concatenate((y1, nan_array, y2))

        # Plot the circle with dotted lines
        self.radius_circle.setData(x, y, pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DotLine))


    def compute_3d_map(self, z=None):
        # #####
        label_noProb = self.map_color.copy()
        label_noProb[:, :, 3] = 0.7
        return self.cur_Iinv.cpu().numpy(), label_noProb
    
    def init_widget_3d(self):
        data3d = self.X[:, :3]
        color_data = self.c_scatter

        X3D_inv, color_surf = self.compute_3d_map()
        # X3D_inv = X3D_inv

        ## create a canvas
        canvas = scene.SceneCanvas(keys='interactive', show=True)
        # canvas.measure_fps()
        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        # view.camera.fov = 50
        # view.camera.distance = 6
        # view.camera.elevation = 30
        # view.camera.azimuth = 30
        # view.camera.scale_factor = 1.0
        view.camera.center = np.mean(data3d, axis=0)
        
        # print(X3D[:, :5])
        # Add a surface plot
        surface = scene.visuals.SurfacePlot(X3D_inv[:,0].reshape(self.GRID,self.GRID), X3D_inv[:,1].reshape(self.GRID, self.GRID), X3D_inv[:,2].reshape(self.GRID, self.GRID))
        # assign surface to scatters 
        # surface = scene.visuals.Markers()
        # surface.set_data(X3D, edge_color='w', face_color=color_surf.reshape(-1, 4), size=5)
        view.add(surface)
        surface.mesh_data.set_vertex_colors(color_surf.reshape(-1, 4))
        # # Add a 3D axis to keep us oriented
        axis = scene.visuals.XYZAxis(parent=view.scene)
        # # Add a colored 3D scatter plot
        scatter = scene.visuals.Markers()
        scatter.set_data(data3d, edge_color='w', face_color=color_data, size=9)
        view.add(scatter)

        ## connect to keyboard
        canvas.events.key_press.connect(self.vispy_key_press)

        ## backgroud color white
        view.bgcolor = (0.3,0.3,0.3, 1)

        return canvas, surface
        

    def init_widget_side(self):
        ## a V layout for side widgets
        ## tops are buttons and bottom is a plot widget
        vlayout = QtWidgets.QVBoxLayout()
        # buttons
        button1 = QtWidgets.QPushButton('Rest to original z')
        # button2 = QtWidgets.QPushButton('Button 2')
        button1.clicked.connect(self.reset_z)


        ##### BLOCK： DISPLAY SETTINGS
        ##### map content combobox
        label_map_content = QtWidgets.QLabel('Map content:')
        combobox_map = QtWidgets.QComboBox()
        content_list = ['None', 'Decision map', 'Distance map with confidence', 'Distance to the initial surface', 'Gradient map', 'Distance to the nearest neighbor']
        combobox_map.addItems(content_list)
        combobox_map.activated.connect(self.onActivated_map)
        ## set default
        combobox_map.setCurrentIndex(1)
        self.map_content = 1

        row_map_content_layout = QtWidgets.QHBoxLayout()
        row_map_content_layout.addWidget(label_map_content, 1)
        row_map_content_layout.addWidget(combobox_map, 2)

        ### data ponit combo box
        label_data_point = QtWidgets.QLabel('Data point:')
        combobox_data_point = QtWidgets.QComboBox()
        combobox_data_point.addItems(['color label', 'color label | size distance','hide data points', 'color distace to the surface'])
        combobox_data_point.activated.connect(self.onActivated_scatter_plot)
        combobox_data_point.setCurrentIndex(0)
        self.show_scatters = True

        slider3 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider3.setRange(1, 50)
        slider3.setValue(self.scatter_size_factor)
        label3 = QtWidgets.QLabel('data ponit size')

        block_display = QtWidgets.QWidget()
        block_display_layout = QtWidgets.QVBoxLayout()

        # row_data_point = QtWidgets.QWidget()
        row_data_point_layout = QtWidgets.QHBoxLayout()
        row_data_point_layout.addWidget(label_data_point, 1)
        row_data_point_layout.addWidget(combobox_data_point, 2)
        # row_data_point.setLayout(row_data_point_layout) 
        
        # row_data_point_slider = QtWidgets.QWidget()
        row_data_point_slider_layout = QtWidgets.QHBoxLayout()
        row_data_point_slider_layout.addWidget(label3)
        row_data_point_slider_layout.addWidget(slider3)
        # row_data_point_slider.setLayout(row_data_point_slider_layout)

        block_display_layout.addLayout(row_map_content_layout)
        block_display_layout.addLayout(row_data_point_layout)
        block_display_layout.addLayout(row_data_point_slider_layout)
        
        block_display.setLayout(block_display_layout)
        ###############################


        
        #### BLOCK: INTERACTION SETTINGS``
        # check box########################################################
        checkbox1 = QtWidgets.QCheckBox('Adjust z by clicking anchors (data ponits)')
        # set status
        checkbox1.toggled.connect(self.shap_status)
        self.shap = False

        

        # slider  ######################################################
        slider0 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider0.setRange(-250, 350)
        # slider0.set
        slider0.setValue(0)
        slider0.setTickInterval(20)
        label0 = QtWidgets.QLabel('z value of above')
        self.slider_z = slider0
        # wrap slider and label in a widget
 
        # Slider1
        slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider1.setRange(0, 100)  # Representing 0 to 1 with a precision of 0.01
        slider1.setValue(50)  # Represents the value 0.5
        slider1.setTickInterval(1)  # Represents the interval 0.01
        label1 = QtWidgets.QLabel('factor')

        # Slider2
        slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider2.setRange(10, 300)  # Representing 0 to 0.5 with a precision of 0.01
        slider2.setValue(self.shape_radius*1000)  # Represents the value 0.25
        slider2.setTickInterval(1)  # Represents the interval 0.01
        label2 = QtWidgets.QLabel('radius (&#963;)')  # Use the HTML code for sigma
        label2.setTextFormat(QtCore.Qt.RichText)  # Enable rich text rendering
        label2.setText('<html><body><span style="font-size:12pt;">radius (&#963;)</span></body></html>')


        # Slider control how far to go with delta z
        slider_control = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_control.setRange(-120, 120)  # Representing -1 to 1 with a precision of 0.01
        slider_control.setValue(0)  # Represents the value 0
        slider_control.setTickInterval(1)  # Represents the interval 0.05
        # Set the QLabel text to display alpha using HTML
        label_control = QtWidgets.QLabel(r'$\alpha$')
        label_control.setTextFormat(QtCore.Qt.RichText)  # Enable rich text rendering
        label_control.setText('<html><body><span style="font-size:12pt;">&#945;</span></body></html>')  # Use the HTML code for alpha

        self.control_silder = slider_control



        # set layout for slider and label

        s0_layout = QtWidgets.QVBoxLayout()
        # s0_layout.addWidget(slider0)
        # s0_layout.addWidget(slider1)
        s0_layout.addWidget(slider2)
        s0_layout.addWidget(slider_control)

 
        s1_layout = QtWidgets.QVBoxLayout()
        # s1_layout.addWidget(label0)
        # s1_layout.addWidget(label1)
        s1_layout.addWidget(label2)
        s1_layout.addWidget(label_control)


        # SL = QtWidgets.QWidget()
        SL_layout = QtWidgets.QHBoxLayout()
        SL_layout.addLayout(s1_layout)
        SL_layout.addLayout(s0_layout)


        # sliders connect 
        slider0.valueChanged.connect(self.slider0_changed)
        slider1.valueChanged.connect(self.slider1_changed)
        slider2.valueChanged.connect(self.slider2_changed)
        slider3.valueChanged.connect(self.slider3_changed)
        slider_control.sliderReleased.connect(self.slider_control_changed)
        self.update_sliderValue_only = False
        ################################################################

        # combo box######################################################
        textForCombox = QtWidgets.QLabel('Control panel:')
        combo_dim = QtWidgets.QComboBox()
        combo_dim.addItem('All Dimensions')
        for i in range(self.current_z.shape[1]):
            combo_dim.addItem(f'dim_of_z {i}')
        combo_dim.activated[int].connect(self.onActivated_dim)
        self.dimChange_slider = 'all_dim'

        combox_local = QtWidgets.QComboBox()
        combox_local.addItem(',globally')
        combox_local.addItem(',locally (centered at "+")')
        combox_local.activated[int].connect(self.locality_status)
        self.locality = False

        # combox_group = QtWidgets.QWidget()
        combox_group_layout = QtWidgets.QHBoxLayout()
        combox_group_layout.addWidget(combo_dim)
        combox_group_layout.addWidget(combox_local)
        # combox_group.setLayout(combox_group_layout)

        block_interaction = QtWidgets.QWidget()
        block_interaction_layout = QtWidgets.QVBoxLayout()
        # block_interaction_layout.addWidget(checkbox1)  ## temporarily hide this
        block_interaction_layout.addWidget(textForCombox) 
        # block_interaction_layout.addLayout(combox_group_layout) ## temporarily hide this
        block_interaction_layout.addLayout(SL_layout)
        block_interaction.setLayout(block_interaction_layout)
       ############################################################### END BLOCK INTERACTION SETTINGS

       ##### input index and save button
        ### edit box
        e1 = QtWidgets.QLineEdit()
        e1.setValidator(QtGui.QIntValidator())
        e1.setMaxLength(4)

        e1.editingFinished.connect(self.enter_index)
        e1_label = QtWidgets.QLabel('Mannual index:')
        save_button = QtWidgets.QPushButton('Undo change')
        save_button.clicked.connect(self.undo_changes)
        block_save = QtWidgets.QWidget()
        block_save_layout = QtWidgets.QHBoxLayout()
        block_save_layout.addWidget(e1_label)
        block_save_layout.addWidget(e1)
        block_save_layout.addWidget(save_button)
        block_save_layout.addWidget(button1)
        block_save.setLayout(block_save_layout)


        ##### BLOCK: real and inverse image display
        block_image = QtWidgets.QWidget()
        block_image_layout = QtWidgets.QVBoxLayout()
        image_only_layout = QtWidgets.QHBoxLayout()
        inverse_onclick = pg.ImageItem(np.zeros(self.data_shape))
        plot_inv = pg.plot()
        # plot_inv.setAspectLocked(True, ratio=1)
        plot_inv.addItem(inverse_onclick)
        # plot_inv.setRect(QtCore.QRectF(0, 0, 1, 1))

        real = pg.ImageItem(np.ones(self.data_shape)*128)
        plot_real = pg.plot()
        # plot_real.setAspectLocked(True, ratio=1)
        ## set ratio
        vb_real = plot_real.getViewBox()
        vb_real.setAspectLocked(True)
        vb_inv = plot_inv.getViewBox()
        vb_inv.setAspectLocked(True)
        ##########

        plot_real.addItem(real)
        # plot_real.setRect(QtCore.QRectF(0, 0, 1, 1))
        plot_inv.setLimits(xMin=0, xMax=self.data_shape[0], yMin=0, yMax=self.data_shape[1])
        plot_real.setLimits(xMin=0, xMax=self.data_shape[0], yMin=0, yMax=self.data_shape[1])
        ## hide axis
        plot_inv.hideAxis('bottom')
        plot_inv.hideAxis('left')
        plot_real.hideAxis('bottom')
        plot_real.hideAxis('left')
        image_only_layout.addWidget(plot_real)
        image_only_layout.addWidget(plot_inv)
        block_image_layout.addLayout(image_only_layout)
        ## a label widget
        image_label_layout = QtWidgets.QHBoxLayout()
        real_label = QtWidgets.QLabel('Real sample (Target)')
        inverse_label = QtWidgets.QLabel('Inverse projection (Source)')
        real_label.setAlignment(QtCore.Qt.AlignCenter)
        inverse_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label_layout.addWidget(real_label)
        image_label_layout.addWidget(inverse_label)
        block_image_layout.addLayout(image_label_layout)
        block_image.setLayout(block_image_layout)

        ############################### END BLOCK real and inverse image display


        ###################### console display
        self.console_list = [
            'Press number key (0-4), then click on the map to set an observation window;',
            'Press Shift and click on a scatter point to select Real Sample;',
            ' ', 
        ]

        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)

        self.console.setText('\n'.join(self.console_list))

        self.console.setFont(QtGui.QFont("Courier", 1))

        # Enable text wrapping at the widget width
        self.console.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)

        # Vertical scrollbar always visible, horizontal scrollbar disabled 
        self.console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.console.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                

        ### score label block 
        self.score_label_accm = QtWidgets.QLabel('Map Accuracy: --')
        self.score_label_cons = QtWidgets.QLabel('Consistency: --')
        self.score_label_accm.setTextFormat(QtCore.Qt.RichText)
        self.score_label_cons.setTextFormat(QtCore.Qt.RichText)
        ## check box for updating the scores
        self.score_label_box = QtWidgets.QCheckBox('Update scores')
        self.score_label_box.toggled.connect(self.update_score_label)
        self.score_label_box.setChecked(True)
        self.update_score = True
        # self.update_the_scores()
        ## set layout for score label
        score_label = QtWidgets.QWidget()
        score_label_vlayout = QtWidgets.QVBoxLayout()
        score_label_layout = QtWidgets.QHBoxLayout()
        score_label_layout.addWidget(self.score_label_accm)
        score_label_layout.addWidget(self.score_label_cons)
        score_label_vlayout.addWidget(self.score_label_box)
        score_label_vlayout.addLayout(score_label_layout)
        score_label.setLayout(score_label_vlayout)

        ## 
        save_z_button = QPushButton('Save current z')
        load_z_button = QPushButton('Load z')  # Use HTML tags for bold text
        save_z_button.clicked.connect(self.save_z)
        load_z_button.clicked.connect(self.load_z)
        layout_save_load = QtWidgets.QHBoxLayout()
        layout_save_load.addWidget(save_z_button)
        layout_save_load.addWidget(load_z_button)
        block_save_load = QtWidgets.QWidget()
        block_save_load.setLayout(layout_save_load)



        # add widgets to layout
        # vlayout.addWidget(button1, )
        vlayout.addWidget(block_save_load)
        vlayout.addWidget(score_label, ) # 1
        vlayout.addWidget(block_display, ) # 3
        vlayout.addWidget(block_interaction, ) # 6
        vlayout.addWidget(block_save, ) # 7
        vlayout.addWidget(block_image,  ) 
        vlayout.addWidget(self.console, )
 
        # vlayout.addWidget(textForCombox)
        # vlayout.addWidget(combox_group)
        # vlayout.addWidget(SL)
    
        # vlayout.addWidget(plot_real)
        # vlayout.addWidget(plot_inv)
        return vlayout, inverse_onclick, real
    
    def reset_z(self):
        self.current_z = self.initial_z.copy()
        self.update_2d_map()
        self.update_inverse_window(self.mouse_pos[0])
        self.control_silder.setValue(0)
        self.cache_source_z = self.get_z(self.mouse_pos)
        if self.show3d:
            self.update_3d()


    def shap_status(self, checked):
        if checked:
            print('Checked')
            self.shap = True
        else:
            self.shap = False
            self.radius_circle.setData(x=[], y=[])
            print('Unchecked')

    def update_score_label(self, checked):
        if checked:
            self.update_score = True
            self.update_the_scores()
        else:
            self.update_score = False
            ## set the text to be gray color
            self.score_label_accm.setText(f'<font color="gray">Map Accuracy: {self.acc_score:.4f} </font>')
            self.score_label_cons.setText(f'<font color="gray">Consistency: {self.cons_score:.4f}</font>')

    def locality_status(self, index):
        if index == 1:
            print('the slider below works _locally_')
            self.locality = True
        else:
            print('the slider below works _globally_')
            self.locality = False

    def get_distance_to_origin(self, update=False):
            # if no change, return 0
            if np.all(self.current_z == self.initial_z):
                print('This is the original map')
                return np.zeros((self.GRID, self.GRID))
            
            # map_diff = self.current_z - self.initial_z
            # map_diff= self.get_inverse(self.XY_grid, z=self.current_z) - self.get_inverse(self.XY_grid, z=self.initial_z)
            if update:
                self.cur_Iinv = self.get_inverse(self.XY_grid, self.current_z, GPU=True)
            map_diff = self.cur_Iinv - self.initial_Iinv  ## isn't this to heavy?

            # L2 norm
            # map_diff = np.linalg.norm(map_diff, axis=1)
            map_diff = torch.norm(map_diff, dim=1).cpu().numpy()
            map_diff = map_diff.reshape(self.GRID, self.GRID)
            ## norm and map to 0-1 ?
            diff_max = np.max(map_diff)
            print('max diff', diff_max)
            map_diff = map_diff / diff_max
            return map_diff
            
    def init_ob_wiget(self):
        vlayout = QtWidgets.QVBoxLayout()
        ob_list = []
        self.obtext_list = []
        for i in range(5):
            ob = pg.ImageItem(np.zeros(self.data_shape))
            ob_widget = pg.plot()
            ob_widget.addItem(ob)
            ob_list.append(ob)
            vlayout.addWidget(ob_widget)
            # omit the axis
            ob_widget.hideAxis('bottom')
            ob_widget.hideAxis('left')

            ## set rect
            ob_widget.setLimits(xMin=0, xMax=self.data_shape[0], yMin=0, yMax=self.data_shape[1])

            label = pg.TextItem(str(i), anchor=(0.5, 0.5), color=[233,233,233])
            ### set the text using html, white facecolor and black stroke
            label.setHtml(f'<div style="color: white; -webkit-text-stroke: 2px black;">{i}</div>')
            self.win1.addItem(label)
            label.setPos(1.5, 1.5)
            # fontsize bigger
            label.setFont(QtGui.QFont("", 23, QtGui.QFont.Bold))
            self.obtext_list.append(label)
            vb = ob_widget.getViewBox()
            vb.setAspectLocked(True)            
        return vlayout, ob_list

        

    def onActivated_map(self, index):
        print(index)
        self.map_content = index
        self.update_2d_map()
         ### no sure 
        if self.show3d:
            print('also update 3d')
            self.update_3d()

    def onActivated_scatter_plot(self, index):
        print(index)
        match index:
            case 0:  # show data points only (no distance)
                print('showing scatter plot')
                self.show_scatters = True
                self.show_scatter_dist = False
                self.ponit_color_mode = 'label'
                sizes = self.process_scatter_size()
                self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.c_scatter*255)
            case 1: # show data points with distance
                self.show_scatters = True
                self.show_scatter_dist = True
                self.ponit_color_mode = 'label'
                print('showing scatters distance changed to:', self.show_scatter_dist)
                self.update_2d_map()

            case 2: # hide data points
                self.show_scatters = False
                self.scatter2d.setData(x=[], y=[])

            case 3: # show data points with color distance
                self.show_scatters = True
                self.show_scatter_dist = False
                self.ponit_color_mode = 'distance'
                self.distances = self.get_distance( GPU=False).astype(np.float16)
                self.color_dist = cm.viridis(self.distances/np.max(self.distances))
                sizes = self.process_scatter_size()
                self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.color_dist*255)

                
       

    def onActivated_dim(self, index):
        print(index)
        if index == 0:
            self.dimChange_slider = 'all_dim'
            # set it to the mean of all z || LEAVE IT FOR NOW
            z_mean = np.mean(self.current_z)
            self.update_slider_z_value_only(value=z_mean*100)
        else:
            self.dimChange_slider = index - 1
            # z_ofclick = self.get_z(self.mouse_pos)
            # # print('z of click', z_ofclick)
            # self.slider_z.setValue(z_ofclick[0, self.dimChange_slider]*100)
            self.update_slider_z_value_only()

    def update_slider_z_value_only(self, value=None):
        self.update_sliderValue_only = True
        if value is None: ## the ckicked 
            if self.mouse_pos is None:
                print('no target point selected')
                return
            z_ofclick = self.get_z(self.mouse_pos)
            self.slider_z.setValue(z_ofclick[0, self.dimChange_slider]*100)
        else:
            self.slider_z.setValue(value)
        self.update_sliderValue_only = False

    def slider0_changed(self, value):
        if self.update_sliderValue_only:
            return

        value_scaled = value / 100  # Compute once and reuse

        if self.locality:
            if self.mouse_pos is None:
                print('no target point selected')
                return

            click_z = self.get_z(self.mouse_pos)[0]

            if self.dimChange_slider == 'all_dim':
                new_z = np.full(self.current_z.shape[1], value_scaled)
            else:
                new_z = click_z.copy()
                new_z[self.dimChange_slider] = value_scaled

            delta_z = new_z - click_z
            delta = delta_z * self.cache_filter
            self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor

        else:
            if self.dimChange_slider == 'all_dim':
                self.current_z = np.full((self.GRID**2, self.current_z.shape[1]), value_scaled)
            else:
                self.current_z[:, self.dimChange_slider] = value_scaled

        self.update_2d_map()
        #
        if self.mouse_pos is not None:
            self.update_inverse_window(self.mouse_pos[0])
        # update 3d widget
        if self.show3d:
            self.update_3d()
        
    def slider1_changed(self, value):
        # print(value)
        self.shape_factor = value/100
       

    def slider2_changed(self, value):
        # print(value)
        self.shape_radius = value/1000
        # if self.data_ind is not None:
        #     self.plot_circle(center=self.X2d[self.data_ind], radius=self.shape_radius)
        if self.mouse_pos is not None:
            self.plot_circle(center=self.mouse_pos[0], radius=self.shape_radius)
            self.cache_filter = self.gaussian_filter(self.mouse_pos[0], sigma=self.shape_radius)
        else:
            self.plot_circle(center=[0.5, 0.5], radius=self.shape_radius)
    

    def slider3_changed(self, value):
        # scatter size
        self.scatter_size_factor = value
        # update
        if self.show_scatter_dist:
            # sizes = (self.distances.max() - self.distances + 0.1*self.distances.max()) * self.scatter_size_factor
            sizes = self.raw_data_sizes * self.scatter_size_factor # 0.2
        else:
            sizes = np.ones(self.X2d.shape[0]) * self.scatter_size_factor * 0.5
        match self.ponit_color_mode:
            case 'label':
                self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.c_scatter*255)
            case 'distance':
                self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.color_dist*255)

    def slider_control_changed(self):
        ## do the calculation only when certain time has passed
        value = self.control_silder.value()
        # time_now = time.time()
        # time_diff = time_now - self.cache_time
        # print('time', time_now)
        # print(f'time diff: {time_diff}')
        # if time_diff < 1:
        #     print('too fast')
        #     return
        # self.cache_time = time_now
        value_scaled = value / 100
        print(f'control slider value: {value_scaled}')
        delta_z = self.cache_target_z - self.cache_source_z
        delta = delta_z * self.cache_filter
        self.current_z = self.cache_z + delta.reshape(-1, self.current_z.shape[1]) * value_scaled
        self.update_2d_map()
        self.update_inverse_window(self.mouse_pos[0])
        if self.show3d:
            self.update_3d()
        


    def update_2d_map(self):
        ## TODO: change it to combobox
        ## TODO: calculate the distance of scatter points only when needed, isolate it from the map calculation
        match self.map_content:
            case 3: # distance to the initial surface
                if self.show3d:
                    map_data = self.compute_2d_map()
                    
                map_diff = self.get_distance_to_origin(update=True)
                color_diff = cm.viridis(map_diff)
                self.map2d.setImage(color_diff)

            case 0: # no map content
                print('no map content')
                # set it to a gray image 
                map_data = np.ones((self.GRID, self.GRID, 4)) * 128
                self.map2d.setImage(map_data)
                
            case 4: # gradient map
                print('gradient map')
                map_data = self.get_gradient_map()
                self.map2d.setImage(map_data)

            case 5:## distance to the enearst data point
                print('distance to the enearst data point')
                map_data = self.get_distance_to_neighbor()
                self.map2d.setImage(map_data)
                

            case _: # decision map, distance map with confidence
                if self.map_content == 1:
                    map_data = self.compute_2d_map(proba=False)
                elif self.map_content == 2:
                    map_data = self.compute_2d_map(proba=True)
                time0 = time.time()
                self.map2d.setImage(map_data)
                print(f'set map2d time: {time.time() - time0}')

        if self.show_scatters or self.update_score:
            if self.show_scatter_dist or self.update_score or self.ponit_color_mode=='distance':
                ## place holder
                ## compute X_reco
                X_rocon = self.compute_X_recon()
                ## upatge score 
                self.update_the_scores(X_recon=X_rocon)
                ## compute sizes
                sizes = self.process_scatter_size(X_recon=X_rocon)
            else:
                sizes = self.process_scatter_size()
            
            match self.ponit_color_mode:
                case 'label':
                    self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.c_scatter*255)
                case 'distance':
                    distance = self.get_distance(X_recon=X_rocon, GPU=False)
                    self.color_dist = cm.viridis(distance/np.max(distance))
                    self.scatter2d.setData(self.X2d[:,0], self.X2d[:,1], symbol='o', size=sizes, brush=self.color_dist*255)
        
        # if self.ob2d_list.all():
        if not self.show3d:
            self.update_ob_windows()

    def update_ob_windows(self):
        ob_z = self.get_z(self.ob2d_list)
        ob_nd_lsit = self.get_inverse(self.ob2d_list, z=ob_z, GPU=False)
        for i in range(len(self.ob_list)):
            if sum(self.ob2d_list[i]) != 2:
                img = ob_nd_lsit[i].reshape(self.data_shape)*255
                img = np.flip(img, axis=0)
                self.ob_list[i].setImage(img)
    
    def update_3d(self):
        if self.show3d == False:
            return
        # update 3d  widget
        X3D_inv, color_surf = self.compute_3d_map() 
        self.surface3d.set_data(X3D_inv[:,0].reshape(self.GRID,self.GRID), X3D_inv[:,1].reshape(self.GRID, self.GRID), X3D_inv[:,2].reshape(self.GRID, self.GRID))
        self.surface3d.mesh_data.set_vertex_colors(color_surf.reshape(-1, 4))
        self.win2.update()
    
    def get_real_data(self, index):
        """
        Overwrite this function when the case is different
        """
    
        if self.X.shape[1] == 3:
            cur_real = self.X[index].reshape(1, 3)
        else:
            cur_real = self.X[index].reshape(self.data_shape)*255
            cur_real = np.flip(cur_real, axis=0)
        return cur_real
    
    def update_real_for_index(self, index):
        self.data_ind = index
        print(f'Current target is index: {index}')
        self.cache_target_z = self.encode(self.X[self.data_ind])
        self.cache_z = self.current_z.copy()
        # self.update_2d_map()
        # if self.show3d:
        #     self.update_3d()
        self.selected_2dPonit.setData(self.X2d[index, 0].reshape(1,-1), self.X2d[index, 1].reshape(1, -1), symbol='o', size=15, brush=None, pen=pg.mkPen(color='r', width=1.5))
        cur_real = self.get_real_data(index)
        self.real_onclick.setImage(cur_real)
        self.control_silder.setValue(0)

    def enter_index(self):
        index = int(self.sender().text())
        if index >= self.X.shape[0]:
            print('index out of range')
            return
        self.update_real_for_index(index)
    # Define callback function for clicked points in scatter plot
    def scatter_clicked(self, plot, points):
        if not self.select_mode:
            return
        for point in points:
            # start_time = time.time()
            x_condition = self.X2d[:, 0] == point.pos().x()
            y_condition = self.X2d[:, 1] == point.pos().y()
            index = np.where(x_condition & y_condition)[0][0]
            self.update_real_for_index(index)
            # start_time = time.time()
            if self.shap:
                self.update_z()
                # print(f'Time taken for updating z: {time.time() - start_time} seconds')
                # start_time = time.time()
                self.update_2d_map()
                if self.show3d:
                    self.update_3d()

            self.update_inverse_window(self.X2d[index])  ### is this redundant? 

            break
    
    def undo_changes(self):
        self.current_z = self.cache_z.copy() ## careful with the .copy()
        self.update_2d_map()
        self.update_inverse_window(self.mouse_pos[0])
        self.control_silder.setValue(0)
        self.cache_source_z = self.get_z(self.mouse_pos)
        print('undo changes')

    # Define callback function for clicked points on the image
    def image_clicked(self, event):
        # print('clicked on image', event)
        print(self.mouse_pos)
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pressed = True
            self.update_inverse_window(event.position())
            if self.ob_ind is not None:
                self.update_inverse_window(event.position(), ob=self.ob_ind)

            if self.ctrl_pressed or self.locality:
                self.plot_circle(center=self.mouse_pos[0], radius=self.shape_radius)


    def image_released(self, event):
        # print('release:', event.button())
        # self.mouse_pressed = False
        if event.button() == QtCore.Qt.LeftButton:
            # print('released on image', event)
            # print('release!!!:', self.mouse_pressed)
            self.mouse_pressed = False
            self.control_silder.setValue(0)
            self.cache_filter = self.gaussian_filter(self.mouse_pos[0], sigma=self.shape_radius)
            self.cache_source_z = self.get_z(self.mouse_pos)
            self.cache_z = self.current_z.copy()  ## save directly || avoid new source but old surface (z)
            # self.z_change_stack.append(self.current_z.copy())
            ## TODO: add cache_z to a stack. then can undo changes

            if self.ctrl_pressed:
                self.plot_circle(center=self.mouse_pos[0], radius=self.shape_radius)
                self.update_z(location=self.mouse_pos[0])
                self.update_2d_map()
                if self.show3d:
                    self.update_3d()

    def image_moved(self, evt):
        # TODO: detect if mouse in the image
        ## 
        # if self.shap: 
        if True:  ## alway show the circle
            viewbox = self.win1.getPlotItem().getViewBox()
            view_coords = viewbox.mapSceneToView(evt)
            mouse_pos = np.array([[view_coords.x(), view_coords.y()]])
            self.plot_circle(center=mouse_pos[0], radius=self.shape_radius)
        else:
            if self.ctrl_pressed or self.locality:
                pass
            else:
                self.radius_circle.setData(x=[], y=[])

        if self.mouse_pressed:  # Only update if mouse is pressed
            # print('updating, moved on image')
            self.update_inverse_window(evt)
            if self.ctrl_pressed or self.locality:
                self.plot_circle(center=self.mouse_pos[0], radius=self.shape_radius)

            # update the slider_z if self.dimChange_slider is not 'all_dim' TODO: or just update when only clicked
            if self.dimChange_slider != 'all_dim':
                self.update_slider_z_value_only()

            
    
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
        cur_inv = self.get_inverse(self.mouse_pos, cur_z, GPU=False)
        # reshape to square 
        if self.show3d:
            cur_inv = cur_inv.reshape(1, 3)
        else:
            cur_inv = cur_inv.reshape(self.data_shape)*255
            cur_inv = np.flip(cur_inv, axis=0)

        
        self.inverse_onclick.setImage(cur_inv) #, axisOrder='row-major')

        if ob is not None: 
            self.ob_list[ob].setImage(cur_inv)
            self.ob_ind = None
            # update ob2d_list
            self.ob2d_list[ob] = self.mouse_pos[0]
 
            # print(self.ob2d_list)

    ## add a key press event

    def save_z(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Set the file dialog to save a file with a default file type and name
        file_filter = "NumPy Files (*.npy);;All Files (*)"
        default_name = "default.npy"

        # Open the file dialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", default_name, file_filter, options=options)

        if file_name:
            print(f"Selected File to Save: {file_name}")
            np.save(file_name, self.current_z)

    def load_z(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Set the file dialog to open a file with a default file type and name
        file_filter = "NumPy Files (*.npy);;All Files (*)"
        default_name = "default.npy"

        # Open the file dialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", default_name, file_filter, options=options)
        if file_name:
            print(f"Selected File to Load: {file_name}")
            self.current_z = np.load(file_name)
            self.update_2d_map()
            self.update_inverse_window(self.mouse_pos[0])
    
    def keyPressEvent(self, event):
        # shift + i
    
        if event.key() == QtCore.Qt.Key_I and event.modifiers() == QtCore.Qt.ShiftModifier:
            print('shift + i pressed')
            if self.cur_init_z % 3 == 0:
                self.z_finder = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1).fit(self.X2d,  self.Pinv.encode(self.X))
                self.initial_z = self.z_finder.predict(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1
                print('initial z by knn')
            elif self.cur_init_z % 3 == 1:
                # self.z_finder = KNeighborsRegressor(n_neighbors=self.X2d.shape[0], weights='distance', n_jobs=-1).fit(self.X2d,  self.Pinv.encode(self.X))
                # self.initial_z = self.z_finder.predict(self.XY_grid)
                interploator = RBFInterpolator(self.X2d, self.Pinv.encode(self.X) , neighbors=None, kernel='linear')
                self.initial_z = interploator(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1      
                print('initial z by rbf linear kernel')      
            else:  
                interploator = RBFInterpolator(self.X2d, self.Pinv.encode(self.X) , smoothing=0.3, neighbors=None)
                self.initial_z = interploator(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1
                print('initial z by smoothed RBF')

        ## shift + r: reset z
        elif event.key() == QtCore.Qt.Key_R and event.modifiers() == QtCore.Qt.ShiftModifier:
            print('shift + r pressed')
            self.reset_z()
        ## shift + c
        elif event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ShiftModifier:
            if self.clf is None:
                self.clf = self.clf_backup
                print('classifier restored')
                self.update_2d_map()
                self.update_3d()
                return
            else:
                self.clf_backup = self.clf
                self.clf = None
                print('classifier removed')
                self.update_2d_map()
                self.update_3d()
                return
        ## 'shift' + ','
        elif event.key() == QtCore.Qt.Key_Comma:
            print('<<<<<-')
            if self.dimChange_slider == 'all_dim':
                self.current_z -= 0.1
            else:
                self.current_z[:, self.dimChange_slider] -= 0.5
                #update slider
                self.update_slider_z_value_only()
            self.update_2d_map()
            #
            if self.mouse_pos is not None:
                self.update_inverse_window(self.mouse_pos[0])
            # update 3d widget
            if self.show3d:
                self.update_3d()

        ## 'shift' + '.'
        elif event.key() == QtCore.Qt.Key_Period:
            print('->>>>>')
            if self.dimChange_slider == 'all_dim':
                self.current_z += 0.1
            else:
                self.current_z[:, self.dimChange_slider] += 0.5
                #update slider
                self.update_slider_z_value_only()

            self.update_2d_map()
            #
            if self.mouse_pos is not None:
                self.update_inverse_window(self.mouse_pos[0])
            # update 3d widget
            if self.show3d:
                self.update_3d()
        ### number keys 0-9
        elif event.key() == QtCore.Qt.Key_0:
            print(event.key())
            self.ob_ind = 0
        elif event.key() == QtCore.Qt.Key_1:
            self.ob_ind = 1
        elif event.key() == QtCore.Qt.Key_2:
            self.ob_ind = 2
        elif event.key() == QtCore.Qt.Key_3:
            self.ob_ind = 3
        elif event.key() == QtCore.Qt.Key_4:
            self.ob_ind = 4
        # elif event.key() == QtCore.Qt.Key_5:
        #     self.ob_ind = 5
        # ctrl
        elif event.key() == QtCore.Qt.Key_Control:
            self.ctrl_pressed = True
            print('ctrl pressed')
        # [ shift
        elif event.key() == QtCore.Qt.Key_Shift:
            self.select_mode = True
            print('shift pressed')
        elif event.key() == QtCore.Qt.Key_BracketLeft:
            print('[ pressed')
            cur_pos = self.mouse_pos[0]
            vert = [[cur_pos[0], cur_pos[1]-0.1], [cur_pos[0], cur_pos[1]+0.1]]
            hori = [[cur_pos[0]-0.1, cur_pos[1]], [cur_pos[0]+0.1, cur_pos[1]]]
            vh = np.array([vert[0], vert[1], hori[0], hori[1], cur_pos])
            vh_z = self.get_z(vh)
            vh_inv = self.get_inverse(vh, vh_z, GPU=True)
            # cross product use torch API 
            h_diff = vh_inv[1] - vh_inv[0]
            v_diff = vh_inv[3] - vh_inv[2]
            cross = torch.cross(h_diff, v_diff, -1) * 100
            print('---------------------------------------debug cross')
            print(cross)
            new_loc = vh_inv[4] + cross
            new_z =  self.Pinv.encode(new_loc)
            delta_z = new_z - vh_z[4]
            print('delta z', delta_z)
            delta = delta_z * self.cache_filter
            self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor
            self.update_2d_map()
            if self.show3d:
                self.update_3d()
        elif event.key() == QtCore.Qt.Key_BracketRight:
            print('] pressed')
            cur_pos = self.mouse_pos[0]
            vert = [[cur_pos[0], cur_pos[1]-0.1], [cur_pos[0], cur_pos[1]+0.1]]
            hori = [[cur_pos[0]-0.1, cur_pos[1]], [cur_pos[0]+0.1, cur_pos[1]]]
            vh = np.array([vert[0], vert[1], hori[0], hori[1], cur_pos])
            vh_z = self.get_z(vh)
            vh_inv = self.get_inverse(vh, vh_z, GPU=True)
            # cross product use torch API 
            h_diff = vh_inv[1] - vh_inv[0]
            v_diff = vh_inv[3] - vh_inv[2]
            cross = torch.cross(h_diff, v_diff, -1) * 100
            print('---------------------------------------debug cross')
            print(cross)
            new_loc = vh_inv[4] - cross
            new_z =  self.Pinv.encode(new_loc)
            delta_z = new_z - vh_z[4]
            print('delta z', delta_z)
            delta = delta_z * self.cache_filter
            self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor
            self.update_2d_map()
            if self.show3d:
                self.update_3d()
        ## ctrl + s
        elif event.key() == QtCore.Qt.Key_S and event.modifiers() == QtCore.Qt.ControlModifier:
            # X_train_unscaled = w_scaler.inverse_transform(X_train)
            # np.savez(f'./cache/{dataset_name}/AFHQv2_train', X_train=X_train, y_train=y_train, X2d_unscaled=X2d_train)
            self.Pinv.save_model(f'./cache/AFHQv2_py')
            x_train = self.X
            X2d_unscaled = self.scaler2d.inverse_transform(self.X2d)
            y_train = self.y
            np.savez(f'./cache/AFHQv2_py/AFHQv2_train.npz', X_train=x_train, y_train=y_train, X2d_unscaled=X2d_unscaled)

        
    
    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.ctrl_pressed = False
            print('ctrl released')
        elif event.key() == QtCore.Qt.Key_Shift:
            self.select_mode = False
            print('shift released')

    def vispy_key_press(self, event):
        print(event.key.name)
        if event.key.name == 'I':
            print('shift + i pressed -- vispy')
            if self.cur_init_z % 3 == 0:
                self.z_finder = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1).fit(self.X2d,  self.Pinv.encode(self.X))
                self.initial_z = self.z_finder.predict(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1
                print('initial z by knn')
            elif self.cur_init_z % 3 == 1:
                # self.z_finder = KNeighborsRegressor(n_neighbors=self.X2d.shape[0], weights='distance', n_jobs=-1).fit(self.X2d,  self.Pinv.encode(self.X))
                # self.initial_z = self.z_finder.predict(self.XY_grid)
                # self.reset_z()
                # self.cur_init_z += 1      
                # print('initial z by knn, all points')     
                interploator = RBFInterpolator(self.X2d, self.Pinv.encode(self.X) , neighbors=None, kernel='linear')
                self.initial_z = interploator(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1
                print('initial z by rbf linear kernel')
            else:  
                interploator = RBFInterpolator(self.X2d, self.Pinv.encode(self.X) , smoothing=0.3, neighbors=None)
                self.initial_z = interploator(self.XY_grid)
                self.reset_z()
                self.cur_init_z += 1
                print('initial z by RBF')
        ## shift + c
        elif event.key.name == 'C':
            if self.clf is None:
                self.clf = self.clf_backup
                print('classifier restored')
                self.update_2d_map()
                self.update_3d()
                return
            else:
                self.clf_backup = self.clf
                self.clf = None
                print('classifier removed')
                self.update_2d_map()
                self.update_3d()
                return
        ## 'shift' + '-'
        elif event.key.name == ',': # and event.modifiers() == QtCore.Qt.ShiftModifier:
            print('<<<<<')
            if self.dimChange_slider == 'all_dim':
                self.current_z -= 0.1
            else:
                self.current_z[:, self.dimChange_slider] -= 0.5
                #update slider
                self.update_slider_z_value_only()
            self.update_2d_map()
            #
            if self.mouse_pos is not None:
                self.update_inverse_window(self.mouse_pos[0])
            # update 3d widget
            if self.show3d:
                self.update_3d()

        ## 'shift' + '+'
        elif event.key.name == '.':
            print('>>>>>')
            if self.dimChange_slider == 'all_dim':
                self.current_z += 0.1
            else:
                self.current_z[:, self.dimChange_slider] += 0.5
                #update slider
                self.update_slider_z_value_only()

            self.update_2d_map()
            #
            if self.mouse_pos is not None:
                self.update_inverse_window(self.mouse_pos[0])
            # update 3d widget
            if self.show3d:
                self.update_3d()
        ## [ and ]
        elif event.key.name == '[':
            print('[ pressed')
            cur_pos = self.mouse_pos[0]
            vert = [[cur_pos[0], cur_pos[1]-0.1], [cur_pos[0], cur_pos[1]+0.1]]
            hori = [[cur_pos[0]-0.1, cur_pos[1]], [cur_pos[0]+0.1, cur_pos[1]]]
            vh = np.array([vert[0], vert[1], hori[0], hori[1], cur_pos])
            vh_z = self.get_z(vh)
            vh_inv = self.get_inverse(vh, vh_z, GPU=True)
            # cross product use torch API 
            h_diff = vh_inv[1] - vh_inv[0]
            v_diff = vh_inv[3] - vh_inv[2]
            cross = torch.cross(h_diff, v_diff) * 100
            print('---------------------------------------debug cross')
            print(cross)
            new_loc = vh_inv[4] + cross
            new_z =  self.Pinv.encode(new_loc)
            delta_z = new_z - vh_z[4]
            print('delta z', delta_z)
            delta = delta_z * self.cache_filter
            self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor
            self.update_2d_map()
            if self.show3d:
                self.update_3d()
        elif event.key.name == ']':
            print('] pressed')
            cur_pos = self.mouse_pos[0]
            vert = [[cur_pos[0], cur_pos[1]-0.1], [cur_pos[0], cur_pos[1]+0.1]]
            hori = [[cur_pos[0]-0.1, cur_pos[1]], [cur_pos[0]+0.1, cur_pos[1]]]
            vh = np.array([vert[0], vert[1], hori[0], hori[1], cur_pos])
            vh_z = self.get_z(vh)
            vh_inv = self.get_inverse(vh, vh_z, GPU=True)
            # cross product use torch API 
            h_diff = vh_inv[1] - vh_inv[0]
            v_diff = vh_inv[3] - vh_inv[2]
            cross = torch.cross(h_diff, v_diff) * 100
            print('---------------------------------------debug cross')
            print(cross)
            new_loc = vh_inv[4] - cross
            new_z =  self.Pinv.encode(new_loc)
            delta_z = new_z - vh_z[4]
            print('delta z', delta_z)
            delta = delta_z * self.cache_filter
            self.current_z += delta.reshape(-1, self.current_z.shape[1]) * self.shape_factor
            self.update_2d_map()
            if self.show3d:
                self.update_3d()





