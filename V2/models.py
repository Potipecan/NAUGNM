import os
import math
import pandas
import torch
from torch.nn import *
from itertools import chain, pairwise

class BasicAutoencoder(torch.nn.Module):
    def __init__(self, input_size, latent_dim, coder_dims):
        super().__init__()
        flat_input = math.prod(input_size)
        self.flatten = Flatten()
        self.unflatten = Unflatten(dim=1, unflattened_size=input_size)

        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
        self.latent = Linear(coder_dims[-1], latent_dim)

        for idx, (i, o) in enumerate(pairwise(chain([flat_input], coder_dims))):
            self.encoder.add_module(f'Linear_{idx}', Linear(i, o))
            self.encoder.add_module(f'RElU_{idx}', ReLU(inplace=True))

        for idx, (i, o) in enumerate(pairwise(chain([latent_dim], coder_dims[::-1], [flat_input]))):
            self.decoder.add_module(f'Linear_{idx}', Linear(i, o))
            self.decoder.add_module(f'RElU_{idx}', ReLU(inplace=True) if idx < len(coder_dims) else Sigmoid())

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        x = self.unflatten(x)
        return x

class VariadicAE(torch.nn.Module):
    def __init__(self, input_size, latent_dim, coder_dims):
        super().__init__()
        flat_input = math.prod(input_size)
        self.flatten = Flatten()
        self.unflatten = Unflatten(dim=1, unflattened_size=input_size)

        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
        self.logvar_layer = Linear(coder_dims[-1], latent_dim)
        self.mean_layer = Linear(coder_dims[-1], latent_dim)
        self.output_mean_std = False

        for idx, (i, o) in enumerate(pairwise(chain([flat_input], coder_dims))):
            self.encoder.add_module(f'Linear_{idx}', Linear(i, o))
            self.encoder.add_module(f'RElU_{idx}', ReLU(inplace=True))

        for idx, (i, o) in enumerate(pairwise(chain([latent_dim], coder_dims[::-1], [flat_input]))):
            self.decoder.add_module(f'Linear_{idx}', Linear(i, o))
            self.decoder.add_module(f'RElU_{idx}', ReLU(inplace=True) if idx < len(coder_dims) else Sigmoid())

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)

        z = torch.randn_like(mean)
        z.mul_(logvar)
        z.add_(mean)

        y = self.decoder(z)
        y = self.unflatten(y)
        if self.output_mean_std:
            return y, mean, logvar
        return y

class BetaKLDivLoss(Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.rec_loss = MSELoss(reduction='sum')

    @staticmethod
    def _kl_div_loss(z_mean, z_logvar):
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar), dim=1)
        kl_loss = kl_loss.mean()
        return kl_loss

    def forward(self, pred, target):
        pred, z_mean, z_logvar = pred
        return self.rec_loss(pred, target) / pred.size(0) + self.beta * BetaKLDivLoss._kl_div_loss(z_mean, z_logvar)

def train(model: Module, device, train_loader, val_loader, optimizer, loss_f, epochs, model_name, model_dir='weights', patience=10000, epsilon=0.0) -> pandas.DataFrame:

    scaler = torch.amp.GradScaler()
    columns = ['loss', 'val_loss']
    rows = []

    best_val_loss = float('inf')
    no_improvement = 0
    last_loss = float('inf')

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        ## train
        model.train()
        tr_loss = 0
        val_loss = 0
        tr_samp_n = 0
        val_samp_n = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = model(x)
            loss = loss_f(y, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
            tr_samp_n += x.size(0)

        tr_loss /= tr_samp_n

        ##val
        model.eval()
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = model(x.to(device, non_blocking=True))
                loss = loss_f(y, x)
                val_loss += loss.item() * x.size(0)
                val_samp_n += x.size(0)

        val_loss /= val_samp_n
        if val_loss < best_val_loss:
            torch.save(model, os.path.join(model_dir, f'{model_name}.best.pth')) # save best
            best_val_loss = val_loss

        rows.append([tr_loss, val_loss])
        print(f'Epoch {epoch + 1}: Train loss: {tr_loss:.6f}, Validation loss: {val_loss}')

        # early stopping
        if abs(val_loss - last_loss) < epsilon:
            no_improvement += 1
            if no_improvement >= patience:
                print(f'Patience exceeded. Stopping on epoch {epoch + 1} ...')
                break
        else:
            no_improvement = 0
            last_loss = val_loss

    torch.save(model, os.path.join(model_dir, f'{model_name}.final.pth'))

    return pandas.DataFrame(rows, columns=columns)

