import torch as T
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F


class NNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        layer_sizes: tuple[int, ...] = (512, 512, 512),
        act: type[nn.Module] = nn.ReLU,
        device = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.classes_ = np.array(range(n_classes))
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if layer_sizes is None:
            self.layers = [nn.Linear(in_features=input_dim, out_features=n_classes)]
        else:
            self.layers = [
                nn.Linear(in_features=i, out_features=o)
                for i, o in zip((input_dim,) + layer_sizes, layer_sizes + (n_classes,))
            ]
        self._act = act

        self.network = nn.Sequential()
        for layer in self.layers[:-1]:
            self.network.append(layer)
            self.network.append(self._act())
        self.network.append(self.layers[-1])
        self.network.to(self.device)


    def forward(self, inputs) -> T.Tensor:
        return F.softmax(self.network(inputs), dim=-1)

    def predict_proba(self, inputs, GPU=False) -> T.Tensor:
        # check if inputs is a numpy array

        if isinstance(inputs, np.ndarray):  
            tensor = torch.from_numpy(inputs).to(self.device)
        elif isinstance(inputs, torch.Tensor):
            if inputs.device != self.device:
                tensor = inputs.to(self.device)
            else:
                tensor = inputs
        else:
            raise ValueError("inputs must be either a numpy array or torch.Tensor")
        with torch.no_grad():
            if GPU and self.device != 'cpu':
                probabilities = self.forward(tensor)
            else:
                probabilities = self.forward(tensor).cpu().numpy()
        return probabilities

    def predict(self, inputs) -> T.Tensor:
        return T.argmax(self.predict_proba(inputs, GPU=True), axis=1).cpu().numpy()

    def activations(self, inputs) -> T.Tensor:
        return self.network(inputs)

    def classify(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[1].squeeze()

    def prob_best_class(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[0]

    def classification_entropy(self, inputs) -> T.Tensor:

        probs = self.forward(inputs)
        assert self.n_classes == probs.size(1)
        # probs = probs[probs > 0]
        entropy = T.where(probs > 0, -T.log(probs) * probs, 0.0)
        return (entropy / T.log(T.tensor(self.n_classes))).sum(dim=1)

    def init_parameters(self):
        self.apply(init_model)

    def fit(self, dataset: Dataset, epochs: int = 150, optim_kwargs: dict = {}):

        train_dl = DataLoader(dataset, batch_size=256, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), **optim_kwargs)
        print(epochs)
        print(type(epochs))
        loop = trange(epochs)
        for e in loop:
            epoch_loss = 0.0
            epoch_n = 0

            for batch in train_dl:
                inputs, targets = batch

                self.zero_grad()
                outputs = self(inputs)

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * targets.size(0)
                epoch_n += targets.size(0)
            loop.set_description(f"Loss: {epoch_loss/epoch_n:.4f}")
        return self
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)

def init_model(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        layer_sizes: tuple[int, ...] = (),
        act: type[nn.Module] = nn.ReLU,
        device = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.classes_ = np.array(range(n_classes))
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        ## compute size of the input to the fully connected layers
        self.input_size = self._get_conv_output(input_dim)

        # conected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=n_classes)

        self.network = nn.Sequential(
            self.conv1,
            F.max_pool2d(kernel_size=2, stride=2),
            self.conv2,
            F.max_pool2d(kernel_size=2, stride=2),
            self.conv3,
            F.max_pool2d(kernel_size=2, stride=2),
            ## view layer
            lambda x: x.view(x.size(0), -1),
            self.fc1,
            F.relu(),
            self.fc2,
            F.relu(),
            self.fc3
        )

        self.network.to(self.device)

    def _get_conv_output(self, shape):
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x.data.view(1, -1).size(1)

    def forward(self, inputs) -> T.Tensor:
        return F.softmax(self.network(inputs), dim=-1)

    def predict_proba(self, inputs, GPU=False) -> T.Tensor:
        # check if inputs is a numpy array

        if isinstance(inputs, np.ndarray):  
            tensor = torch.from_numpy(inputs).to(self.device)
        elif isinstance(inputs, torch.Tensor):
            if inputs.device != self.device:
                tensor = inputs.to(self.device)
            else:
                tensor = inputs
        else:
            raise ValueError("inputs must be either a numpy array or torch.Tensor")
        with torch.no_grad():
            if GPU and self.device != 'cpu':
                probabilities = self.forward(tensor)
            else:
                probabilities = self.forward(tensor).cpu().numpy()
        return probabilities

    def predict(self, inputs) -> T.Tensor:
        return T.argmax(self.predict_proba(inputs, GPU=True), axis=1).cpu().numpy()

    def activations(self, inputs) -> T.Tensor:
        return self.network(inputs)

    def classify(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[1].squeeze()

    def prob_best_class(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[0]

    def classification_entropy(self, inputs) -> T.Tensor:

        probs = self.forward(inputs)
        assert self.n_classes == probs.size(1)
        # probs = probs[probs > 0]
        entropy = T.where(probs > 0, -T.log(probs) * probs, 0.0)
        return (entropy / T.log(T.tensor(self.n_classes))).sum(dim=1)

    def init_parameters(self):
        self.apply(init_model)

    def fit(self, dataset: Dataset, epochs: int = 150, optim_kwargs: dict = {}):

        train_dl = DataLoader(dataset, batch_size=128, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), **optim_kwargs)
        print(epochs)
        print(type(epochs))
        loop = trange(epochs)
        for e in loop:
            epoch_loss = 0.0
            epoch_n = 0

            for batch in train_dl:
                inputs, targets = batch

                self.zero_grad()
                outputs = self(inputs)

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * targets.size(0)
                epoch_n += targets.size(0)
            loop.set_description(f"Loss: {epoch_loss/epoch_n:.4f}")
        return self
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)