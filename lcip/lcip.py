import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import joblib
import os

class PPinv_wrapper:
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

    def fit(self, x, y=None, clf=None):
        self.X2d = self.P.fit_transform(x)
        self.Pinv.fit(self.X2d, x )
        return self


class Encoder(nn.Module):
    """
    The encoder network to encode the input data to latent space z
    """
    def __init__(self, input_dim, hidden_dim=32, z_dim=3, layers=None, use_BN=False):
        super(Encoder, self).__init__()
        if layers is None:
            layers = [hidden_dim*4, hidden_dim*2, hidden_dim]
        self.network = nn.Sequential()
        if use_BN:
            for i, (in_dim, out_dim) in enumerate(zip([input_dim]+layers[:-1], layers)):
                self.network.add_module('linear'+str(i), nn.Linear(in_dim, out_dim))
                self.network.add_module('bn'+str(i), nn.BatchNorm1d(out_dim))
                self.network.add_module('relu'+str(i), nn.ReLU())
            self.network.add_module('linear'+str(len(layers)), nn.Linear(layers[-1], z_dim))
        else:
            for i, (in_dim, out_dim) in enumerate(zip([input_dim]+layers[:-1], layers)):
                self.network.add_module('linear'+str(i), nn.Linear(in_dim, out_dim))
                self.network.add_module('relu'+str(i), nn.ReLU())
        self.network.add_module('linear'+str(len(layers)), nn.Linear(layers[-1], z_dim))
                

    def forward(self, x):
        return self.network(x)
    
        

class Decoder(nn.Module):
    '''
    Takes in z and x2d, output x
    Also the Pinv
    '''
    def __init__(self, output_dim, hidden_dim=128, z_dim=13, layers=None, weight_y=1, use_BN=False):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.dim_expand_y = int(z_dim * weight_y)
        if layers is None:
            layers = [hidden_dim, hidden_dim*2, hidden_dim*4, hidden_dim*8]
        if self.dim_expand_y > 2:
            self.linear0 = nn.Linear(2, z_dim)  ### TODO: adjust the weight of y
            self.linear1 = nn.Linear(z_dim*2, layers[0])
        else:
            self.linear1 = nn.Linear(z_dim+2, layers[0])
        self.network = nn.Sequential()
        if use_BN:
            for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
                self.network.add_module('linear'+str(i), nn.Linear(in_dim, out_dim))
                self.network.add_module('bn'+str(i), nn.BatchNorm1d(out_dim))
                self.network.add_module('relu'+str(i), nn.ReLU())
        else:
            for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
                self.network.add_module('linear'+str(i), nn.Linear(in_dim, out_dim))
                self.network.add_module('relu'+str(i), nn.ReLU())
        self.network.add_module('linear'+str(len(layers)), nn.Linear(layers[-1], output_dim))

    def forward(self, x2d, z):
        if self.dim_expand_y > 2:
            y_in_dimz = self.linear0(x2d)
            y_in_dimz = F.relu(y_in_dimz)
            input_decoder = torch.cat((y_in_dimz, z), dim=1)
        else:
            input_decoder = torch.cat((x2d, z), dim=1)
        hidden = F.relu(self.linear1(input_decoder))
        predicted = self.network(hidden)
        predicted = torch.sigmoid(predicted)
        return predicted
    
    def transform(self, x2d, z):
        return self.forward(x2d, z)


    
class Disentangler(nn.Module):
    """
    A simple neural network to disentangle the latent space z from x2d
    """
    def __init__(self, z_dim,  hidden_dim=128):
        super(Disentangler, self).__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        hidden = F.relu(self.linear2(hidden))
        # output = torch.sigmoid(self.linear3(hidden)) ## if use BCE loss
        output = self.linear3(hidden)
        return output

class AE(nn.Module):
    """
    The autoencoder network
    """
    def __init__(self, input_dim, hidden_dim=32, z_dim=3, layers_e=None, layers_d=None,  weight_y=0, use_BN=False):
        super(AE, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, z_dim, layers=layers_e, use_BN=use_BN)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim, layers=layers_d, weight_y=weight_y, use_BN=use_BN)
     
    def forward(self, x, x2d=None):
        z = self.encoder(x)
        predicted = self.decoder(x2d, z)
        return predicted, z

    def decode(self, x2d=None, z=None):
        predicted = self.decoder.transform(x2d, z)
        return predicted
    


def main_loss(predicted, x, pred_2d=None, true_2d=None, dist_loss=None, beta=1):
    # BCE = F.binary_cross_entropy(predicted, x)
    recon_loss = F.mse_loss(predicted, x)
    ##################################################################
    if pred_2d is not None:
        adv_loss = - dist_loss(pred_2d, true_2d)
        return  recon_loss + beta * adv_loss # + 0.01 * KLD # to do : weight for each loss
    else:
        return recon_loss 


    
##### LCIP 
class LCIP:
    def __init__(self,  hidden_dim=128, z_dim=16, z_neighbor=10, mini_epochs=4, beta=0.1, z_finder_method='rbf', model=None, layers_e=None, layers_d=None,  weight_y=0, use_BN=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('LCIP is using device: ', self.device)
        self.model = model
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.z_neighbor = z_neighbor
        self.mini_epochs = mini_epochs
        self.beta = beta
        self.z_finder_method = z_finder_method
        self.layers_e = layers_e
        self.layers_d = layers_d
        self.weight_y = weight_y
        self.use_BN = use_BN

    def create_model(self, input_dim, z_dim, hidden_dim=32):
        self.model = AE(input_dim, hidden_dim, z_dim=self.z_dim, layers_e=self.layers_e, layers_d=self.layers_d, weight_y=self.weight_y, use_BN=self.use_BN).to(self.device)
        self.Dist = Disentangler(z_dim=z_dim).to(self.device)
        self.optimizerAE = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizerDist = torch.optim.Adam(self.Dist.parameters(), lr=1e-3)
        self.loss_vae = main_loss
        self.loss_Dist = nn.MSELoss()


    def fit(self, x2d, x, y=None, epochs=150, batch_size=128, verbose=0, early_stop=False, patience=5, refit=True, cv_size=0.1):
        ### defualt no continue training |
        if refit:
            self.reset()
        ######################
        
        self.scaler = MinMaxScaler()
        x2d = self.scaler.fit_transform(x2d)
        X2d = torch.from_numpy(x2d).to(self.device)
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).to(self.device)
        else:
            x = x.to(self.device)

        if self.model is None:
            self.create_model(x.shape[1], self.z_dim, self.hidden_dim,)

        if early_stop:
            X_train, X_val, X2d_train, X2d_val = train_test_split(x, x2d, test_size=cv_size, random_state=42)
            X_val = X_val.to(self.device)
            X2d_val_GPU = torch.from_numpy(X2d_val).to(self.device)
            X_train = X_train.to(self.device)
            X2d_train = torch.from_numpy(X2d_train).to(self.device)
        else:
            X_train = x
            X2d_train = X2d
        
        print('size of training data: ', X_train.shape)

        dataset = torch.utils.data.TensorDataset(X_train, X2d_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        min_loss = 1e8
        loss_list = []
        val_loss_list = []
        count = 0
        for epoch in tqdm(range(epochs)):
            epoch_loss = []
            for batch_x, batch_x2d in dataloader:
                if self.mini_epochs > 0: ## use adv loss
                    predicted, z = self.model(batch_x, batch_x2d) ### Do not need to call it multiple times 
                    z_detach = z.detach()
                    for _ in range(self.mini_epochs):
                        self.optimizerDist.zero_grad()
                        pred_2d = self.Dist(z_detach)
                        ## update Dist # could be multiple times
                        loss_Dist = self.loss_Dist(pred_2d, batch_x2d)
                        loss_Dist.backward(retain_graph=True)
                        self.optimizerDist.step()

                ## update VE
                self.optimizerAE.zero_grad()
                predicted, z = self.model(batch_x, batch_x2d) 
                pred_2d = self.Dist(z)

                if self.mini_epochs > 0: ## use adv loss
                    loss_main = self.loss_vae(predicted, batch_x, pred_2d, batch_x2d, self.loss_Dist, self.beta)
                else:
                    loss_main = self.loss_vae(predicted, batch_x, None, None, None, self.beta)
          
                loss_main.backward()
                self.optimizerAE.step()
                epoch_loss.append(loss_main.item())
                loss_list.append(loss_main.item())

            if early_stop:
                # mse of validation set
                with torch.no_grad():
                    ############################3
                    z_val_enconded = self.model.encoder(X_val)
                    predicted_val = self.model.decode(x2d=X2d_val_GPU, z=z_val_enconded)
                    # mse of predicted_val and X_val
                    val_loss = torch.mean((predicted_val - X_val)**2)
                    val_loss_list.append(val_loss.item())
                    if val_loss < min_loss:
                        min_loss = val_loss
                        count = 0
                    else:
                        count += 1
                        if count > patience:
                            print('early stop at epoch: ', epoch)
                            break

            if verbose == 1:
                print("Epoch %d, loss=%.4f" % (epoch, np.mean(epoch_loss)))

        print('stop at epoch: ', epoch)
        
        self.model.eval()
        self.encoded = self.encode(x)
        self.x2d = x2d
        self.fit_zfinder(x2d, self.encoded, method=self.z_finder_method)
        return self
    
    def fit_zfinder(self, x2d, z, method='rbf', k=4):
        if method == 'rbf':
            self.z_finder = RBFInterpolator(x2d, z , smoothing=0.5, neighbors=None)
        elif method == 'knn':
            self.z_finder = KNeighborsRegressor(n_neighbors=self.z_neighbor, weights='distance', n_jobs=-1).fit(x2d, z)
        elif method == 'user':
            self.z_finder = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1).fit(x2d, z)
    
    def transform(self, x2d, z=None, GPU=False, need_scale=True, **kwargs):
        if need_scale:
            x2d = self.scaler.transform(x2d)
        if z is None:
            z = self.find_z(x2d).astype(np.float32)
            z = torch.from_numpy(z).to(self.device)
        else:
            z = torch.tensor(z, dtype=torch.float32).to(self.device)
        x2d = torch.tensor(x2d, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predicted = self.model.decode(x2d, z)
        if GPU:
            return predicted.detach()
        else:
            return predicted.cpu().detach().numpy()
    
    def inverse_transform(self, x2d, z=None, GPU=False, need_scale=True, **kwargs):
        return self.transform(x2d, z, GPU, need_scale, **kwargs)
    
    def encode(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        with torch.no_grad(): 
            z = self.model.encoder(x)
        return z.cpu().numpy()
    
    def find_z(self, x2d, need_scale=False):
        """
        NB: it assumes that x2d is already scaled to [0, 1]
        """
        if need_scale:
            x2d = self.scaler.transform(x2d)

        if self.z_finder_method == 'rbf':
            z = self.z_finder(x2d)
        elif self.z_finder_method == 'knn':
            z = self.z_finder.predict(x2d)
        elif self.z_finder_method == 'user':
            z = self.z_finder.predict(x2d)
        return z
    
    def reset(self):
        self.model = None

    def save_model(self, path='cache/temp'):
        ## if not exist, create folder
        if not os.path.exists(path):
            os.makedirs(path)
        ## save model as pkl
        joblib.dump(self.model, f'{path}/model.pkl')
        ### save scaler
        joblib.dump(self.scaler, f'{path}/scaler2d.pkl')
        ## save z_finder by saving X2d and encoded
        np.save(f'{path}/X2d_scaled.npy', self.x2d)
        np.save(f'{path}/encoded.npy', self.encoded)
        print('model saved at: ', path)

    def load_model(self, path, z_finder_method='rbf'):
        self.model = joblib.load(f'{path}/model.pkl')
        self.model.eval()
        self.model.to(self.device)
        self.scaler = joblib.load(f'{path}/scaler2d.pkl')
        self.x2d = np.load(f'{path}/X2d_scaled.npy')
        self.encoded = np.load(f'{path}/encoded.npy')
        self.fit_zfinder(self.x2d, self.encoded, method=z_finder_method)
        print('model loaded from: ', path)

    def load_z_user(self, path, padding=0.1, k=4):
        xx, yy = np.meshgrid(np.linspace(0-padding, 1+padding, 100), np.linspace(0-padding, 1+padding, 100))
        XY = np.c_[xx.ravel(), yy.ravel()]
        z = np.load(path)
        self.z_finder_method = 'user'
        self.fit_zfinder(XY, z, method='user', k=k)



