from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append('../InverseProjections/')

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from lcip import LCIP
from umap import UMAP
from classifiers import NNClassifier

from sklearn.manifold import TSNE


import torch as T

from tqdm import tqdm






# # MNIST
# (X, y), (_, _) = mnist.load_data()
# X = X.reshape(-1, 28*28)
# X = X.astype('float32') / 255.
X = np.load('./datasets/mnist/X.npy').astype('float32')
y = np.load('./datasets/mnist/y.npy')
# train test split
X_MNIST, _, y_MNIST, _ = train_test_split(X, y, train_size=5000, test_size=2000, random_state=420)
scaler = MinMaxScaler()
X_MNIST = scaler.fit_transform(X_MNIST)

# # fashion MNIST
# (X, y), (_, _) = fashion_mnist.load_data()
# X = X.reshape(-1, 28*28)
# X = X.astype('float32') / 255.
X = np.load('./datasets/fashionmnist/X.npy').astype('float32')
y = np.load('./datasets/fashionmnist/y.npy')
# train test split
X_fashion, _, y_fashion, _ = train_test_split(X, y, train_size=5000, test_size=2000, random_state=420)
scaler = MinMaxScaler()
X_fashion = scaler.fit_transform(X_fashion)

## HAR 
X_har_train = np.load('./datasets/har/X.npy').astype('float32')
# X_har_test = np.load('../sdbm/data/har/X_test.npy').astype('float32')
# X_har = np.concatenate([X_har_train, X_har_test], axis=0)
# X_har = X_har_train
y_har_train = np.load('./datasets/har/y.npy')
# y_har_test = np.load('../sdbm/data/har/y_test.npy')
# y_har = np.concatenate([y_har_train, y_har_test], axis=0)
# y_har = y_har_train
X_har_train, X_har_test, y_har_train, y_har_test = train_test_split(X_har_train, y_har_train, train_size=5000, test_size=2000, random_state=420)

w = np.load('./datasets/w_afhqv2/X.npy')
y = np.zeros(w.shape[0])
y[5065 : -4593] = 1
y[-4593: ] = 2

w_scaler = MinMaxScaler()
w = w_scaler.fit_transform(w)
# split the data
w_train, w_test, y_train, y_test = train_test_split(w, y, train_size=5000, test_size=2000, random_state=420)

X_dict = {
    'AFHQv2': (w_train, y_train),
    'HAR': (X_har_train, y_har_train),
    
    # # 'BLOB': X_blob,
    'MNIST': (X_MNIST, y_MNIST),
    
    'FashionMNIST': (X_fashion, y_fashion),
        
        }

P_dict = {
    'UMAP': UMAP(n_components=2, random_state=42),
            'tSNE': TSNE(n_components=2, random_state=420, n_jobs=-1),
            }

Pinv_name = "LCIP"



def get_results_stacked(lcip, X, X2d, n_surf=50, batch_size=100, padding=0.05, grid=100, clf=None):

    DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
    # get the results
    scaler2d = MinMaxScaler()  ## check later
    X_2d_scaled = scaler2d.fit_transform(X2d)
    
    #### dense data
    x_min, x_max = X_2d_scaled[:, 0].min() - padding, X_2d_scaled[:, 0].max() + padding 
    y_min, y_max = X_2d_scaled[:, 1].min() - padding, X_2d_scaled[:, 1].max() + padding

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid),
                            np.linspace(y_min, y_max, grid))  ## could add more padding
    XY_sample = np.c_[xx.ravel(), yy.ravel()]
    XY_unscaled = scaler2d.inverse_transform(XY_sample)


    ## random sample n_surf from X
    if n_surf < X.shape[0]:
        target_idx = np.random.choice(X.shape[0], n_surf, replace=False)
        X_targets = X[target_idx]
    else:
        X_targets = X
    Zs = lcip.encode(X_targets)

    var_list = []
    label_list = []
    conf_list = []
    for i in tqdm(range(0, XY_unscaled.shape[0], batch_size)):
        XY_batch = XY_unscaled[i:i+batch_size]
        local_surf_stack = T.zeros((XY_batch.shape[0], n_surf, X.shape[1])).to(DEVICE)
        for i, z in enumerate(Zs):
            ## stack z to the batch size as numpy array
            z_repeat = np.stack([z]*XY_batch.shape[0])
            # z_repeat = T.stack([z]*XY_batch.shape[0])
            local_surf = lcip.inverse_transform(XY_batch, z_repeat, GPU=True)
            local_surf_stack[:, i, :] = local_surf
        ## get the variance for each point in the batch shape=(XY_batch)
        local_var = T.var(local_surf_stack, dim=1).sum(dim=1).cpu().detach().numpy()
        var_list.extend(local_var)
        # print(local_var.shape)
        if clf:
            local_surf_flat = local_surf_stack.view(-1, X.shape[1])

            local_conf = clf.predict_proba(local_surf_flat).max(axis=1)
            # print(local_conf.shape)
            local_labels = clf.predict(local_surf_flat)
            

            local_conf = local_conf.reshape(XY_batch.shape[0], n_surf)
            local_labels = local_labels.reshape(XY_batch.shape[0], n_surf)

            conf_list.extend(local_conf)
            label_list.extend(local_labels)
            # print('shape of conf and label')
            # print(local_labels.shape)
            # print(local_conf.shape)
    
    var_list = np.array(var_list)
    ## save 
    # np.save(f'./results/{dataset_name}_{P_name}_{Pinv_name}_{grid}_{padding}_var.npy', var_list)

    if clf:
        ### get inital static surf 
        nd_grid = lcip.inverse_transform(XY_unscaled, GPU=True)
        labels_sinle = clf.predict(nd_grid)
        conf_single = clf.predict_proba(nd_grid).max(axis=1)

        local_conf = np.array(conf_list)
        local_labels = np.array(label_list)
        print('shape of conf and label')
        print(local_labels.shape)
        print(local_conf.shape)

        ## save npz
        np.savez(f'./results/{dataset_name}_{P_name}_{Pinv_name}_{grid}_{padding}.npz', conf=local_conf, label=local_labels, var=var_list, single_conf=conf_single, single_label=labels_sinle)
    else:
        np.savez(f'./results/{dataset_name}_{P_name}_{Pinv_name}_{grid}_{padding}.npz', var=var_list)
        




for dataset_name, (X, y) in X_dict.items():
    clf = NNClassifier(input_dim=X.shape[1], n_classes=np.unique(y).shape[0], layer_sizes=(512, 256, 128))
    dataset = T.utils.data.TensorDataset(T.from_numpy(X).to(clf.device), T.from_numpy(y).long().to(clf.device))
    clf.fit(dataset, epochs=150)
    print("training set acc: ", clf.score(X, y))
    ## save the classifier
    T.save(clf.state_dict(), f'./results/{dataset_name}_clf.pth')

    for P_name, P in P_dict.items():


        X2d = P.fit_transform(X)


        X_train, X2d_train = X, X2d

        if dataset_name == 'AFHQv2':
            licp = LCIP(beta=0.01, mini_epochs=5)
        elif dataset_name == 'HAR':
            licp = LCIP(beta=0.05, mini_epochs=5)
        else:
            licp = LCIP(beta=0.1, mini_epochs=5)
        licp.fit(X2d_train, X_train, batch_size=256, epochs=150)

        print(f'{dataset_name}_{P_name}_{Pinv_name} trained')

        get_results_stacked(licp, X_train, X2d_train, n_surf=5000, batch_size=100, padding=0.05, grid=150, clf=clf)

        ## save X2d and y
        np.save(f'./results/{dataset_name}_{P_name}_{Pinv_name}_X2d.npy', X2d_train)
    np.save(f'./results/{dataset_name}_{Pinv_name}_y.npy', y)
    ## save X
    np.save(f'./results/{dataset_name}_{Pinv_name}_X.npy', X)



    