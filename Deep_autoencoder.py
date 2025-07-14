###############
# This code is adapted from Laurens van der Maaten's autoencoder (Matlab version) code and simplified in certain parts
# to generate Python version autoencoder dimension reduction code.
###############


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random


# 1. Set random seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 2. RBM model definition
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.empty(n_vis, n_hid))
        nn.init.xavier_uniform_(self.W)

        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        _, h_sample = self.sample_h(v)
        v_recon_prob, _ = self.sample_v(h_sample)
        return v_recon_prob

    def contrastive_divergence(self, v, lr=0.01):
        h_prob, h_sample = self.sample_h(v)
        v_recon_prob, _ = self.sample_v(h_sample)
        h_recon_prob, _ = self.sample_h(v_recon_prob)

        pos_grad = torch.matmul(v.t(), h_prob)
        neg_grad = torch.matmul(v_recon_prob.t(), h_recon_prob)

        self.W.data += lr * (pos_grad - neg_grad) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - v_recon_prob, dim=0)
        self.h_bias.data += lr * torch.mean(h_prob - h_recon_prob, dim=0)


# 3. Pretraining stage, train the RBM stack layer by layer, and initialize the autoencoder weights.
def pretrain_autoencoder(data, no_dims=2, layer_sizes=None, epochs=10, batch_size=256, lr=0.01, seed=42):
    set_seed(seed)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    input_size = data.shape[1]
    if layer_sizes is None:
        layer_sizes = [128, 64]
    sizes = [input_size] + layer_sizes + [no_dims]

    rbms = []
    v = torch.tensor(data, dtype=torch.float32)
    for i in range(len(sizes) - 1):
        rbm = RBM(sizes[i], sizes[i+1])
        for epoch in range(epochs):
            idx = torch.randperm(v.size(0))
            for b in range(0, v.size(0), batch_size):
                batch_idx = idx[b:b+batch_size]
                batch = v[batch_idx]
                rbm.contrastive_divergence(batch, lr)
        _, v = rbm.sample_h(v)
        rbms.append(rbm)
    return rbms, scaler


# 4. Autoencoder structure
class Autoencoder(nn.Module):
    def __init__(self, rbms):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential()
        for i, rbm in enumerate(rbms):
            layer = nn.Linear(rbm.W.shape[0], rbm.W.shape[1])
            layer.weight.data = rbm.W.data.t().clone()
            layer.bias.data = rbm.h_bias.data.clone()
            self.encoder.add_module(f"enc_{i}", layer)
            # Encode layers linearly, use Sigmoid for the rest
            if i < len(rbms) - 1:
                self.encoder.add_module(f"act_enc_{i}", nn.Sigmoid())

        self.decoder = nn.Sequential()
        for i, rbm in reversed(list(enumerate(rbms))):
            layer = nn.Linear(rbm.W.shape[1], rbm.W.shape[0])
            layer.weight.data = rbm.W.data.clone()
            layer.bias.data = rbm.v_bias.data.clone()
            self.decoder.add_module(f"dec_{i}", layer)
            self.decoder.add_module(f"act_dec_{i}", nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 5. Fine-tuning stage
def finetune_autoencoder(data, rbms, scaler, epochs=20, batch_size=256, lr=0.001, seed=42):
    set_seed(seed)
    data = scaler.transform(data)
    x = torch.tensor(data, dtype=torch.float32)
    model = Autoencoder(rbms)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        idx = torch.randperm(x.size(0))
        for b in range(0, x.size(0), batch_size):
            batch_idx = idx[b:b+batch_size]
            batch = x[batch_idx]
            encoded, decoded = model(batch)
            loss = loss_fn(decoded, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# 6. Obtain dimension-reduced data
def get_encoded_data(model, data, scaler):
    data = scaler.transform(data)
    x = torch.tensor(data, dtype=torch.float32)
    encoded, _ = model(x)
    return encoded.detach().numpy()
