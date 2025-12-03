import os

import pandas
import torch
import matplotlib.pyplot as plt
from itertools import chain
from typing import Union, Literal
from pathlib import Path

def load_models(*model_names,
                kind: Union[str, Literal['best', 'final']] = 'best',
                device: Union[str, Literal['cpu', 'cuda']] = 'cpu',
                model_dir: Union[str, Path] = 'weights'):
    return {mn: torch.load(os.path.join(model_dir, f'{mn}.{kind}.pth'), weights_only=False, map_location=device) for mn in model_names}

def plot_history(model_name, history_dir = 'histories'):
    df = pandas.read_csv(os.path.join(history_dir, f'{model_name}.history.csv'))
    plot = df[['loss', 'val_loss']].plot()
    plot.set_xlabel('Epoch')
    plot.set_ylabel('Loss')

def show_examples(images: torch.Tensor,
                  models: dict,
                  device: Union[str, Literal['cpu', 'cuda']] = 'cpu'):
    images = images.to(device=device)
    fig, axes = plt.subplots(images.size(0), len(models) + 1, sharex=True, sharey=True)
    for ax_col, (name, model) in zip(axes.T, chain([('Original', lambda x: x)], models.items())):
        ax_col[0].set_title(name)
        with torch.no_grad():
            imgs = model(images)
        for ax, img in zip(ax_col, imgs):
            ax.imshow(img.numpy().squeeze(), cmap='gray')
            ax.set_axis_off()
    fig.tight_layout()
