import os

import pandas
import torch
import matplotlib.pyplot as plt
from itertools import chain
from typing import Union, Literal
from pathlib import Path

from V2.models import VariadicAE


def load_models(*model_names,
                kind: Union[str, Literal['best', 'final']] = 'best',
                device: Union[str, Literal['cpu', 'cuda']] = 'cpu',
                model_dir: Union[str, Path] = 'weights'):
    return {mn: torch.load(os.path.join(model_dir, f'{mn}.{kind}.pth'), weights_only=False, map_location=device) for mn in model_names}

def plot_history(model_name, history_dir = 'histories', ax=None):
    df = pandas.read_csv(os.path.join(history_dir, f'{model_name}.history.csv'))
    plot = df[['loss', 'val_loss']].plot(ax=ax)
    plot.set_title(model_name)
    plot.set_xlabel('Epoch')
    plot.set_ylabel('Loss')

def show_examples(images: torch.Tensor,
                  models: dict,
                  device: Union[str, Literal['cpu', 'cuda']] = 'cpu'):
    images = images.to(device=device)
    fig, axes = plt.subplots(images.size(0), len(models) + 1, sharex=True, sharey=True)
    for ax_col, (name, model) in zip(axes.T, chain([('Original', lambda x: x)], models.items())):
        if isinstance(model, VariadicAE):
            model.output_mean_std = False
        ax_col[0].set_title(name)
        with torch.no_grad():
            imgs = model(images)
        for ax, img in zip(ax_col, imgs):
            img = img.permute(1, 2, 0).squeeze()
            ax.imshow(img.numpy(), cmap='gray' if images.size(1) == 1 else None)
            ax.set_axis_off()
    fig.tight_layout()
