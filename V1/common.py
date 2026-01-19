from itertools import chain
from typing import Literal

import keras
import keras.layers as layers
import pandas
import pandas as pd
from keras.datasets import imdb
import numpy as np
import torchinfo
import matplotlib.pyplot as plt
from pandas import DataFrame
from pathlib import Path

history_dir = Path('history')


def vectorize(seq, dim):
    res = np.zeros((len(seq), dim))  # matrika oblike (len(seq), dim)
    for i, s in enumerate(seq):
        res[i, s] = 1.0  # nastavi indekse na 1
    return res

def load_dataset(n_words=3000):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(cache_dir='imdb/imdb.npz', num_words=n_words)
    x_train = vectorize(train_data, n_words)
    y_train = np.asarray(train_labels, dtype=np.float32)
    x_test = vectorize(test_data, n_words)
    y_test = np.asarray(test_labels, dtype=np.float32)
    return x_train, y_train, x_test, y_test

def load_results(model_name, suffix: Literal['history', 'test'] = 'history'):
    df = pandas.read_csv(history_dir / f'{model_name}.{suffix}.csv')
    return df

def plot_results(*histories, suffix: Literal['history', 'test'] ='history', metrics=('loss', 'accuracy'), metric_prefix=('val',), learning_rates=(1e-3,)):
    
    hdf = [load_results(h, suffix) for h in histories]

    fig, axes = plt.subplots(len(metrics), len(histories), sharey='row')
    if axes.ndim < 2:
        axes = axes[:, np.newaxis]
    for h, mn, col in zip(hdf, histories, axes.T):
        col[0].set_title(mn)
        col[-1].set_xlabel("epoch")

        for m, ax in zip(metrics, col):
            cols = [f'{m} LR={lr:.0e}' for lr in learning_rates] + [f'{p}_{m} LR={lr:.0e}' for p in metric_prefix for lr in learning_rates]
            try:
                plot = h[cols].plot(ax=ax)
            except KeyError:
                cols = [m] + [f'{p}_{m}' for p in metric_prefix]
                plot = h[cols].plot(ax=ax)
            plot.set_ylabel(m)

    plt.tight_layout()
    plt.show()

def init_model(input_dim, units, dropout_rate = 0.0, reg = None):    
    model_name = f'model_{input_dim}_{units}'
    if reg is not None:
        model_name += '_l2'
    l = [
        layers.Input((input_dim,)),
        layers.Dense(units, activation='relu', kernel_regularizer=reg),
        layers.Dense(units, activation='relu', kernel_regularizer=reg),
        layers.Dense(1, activation='sigmoid', kernel_regularizer=reg),
    ]
    
    if dropout_rate > 0:
        model_name += f'_do{dropout_rate}'
        l.insert(3, layers.Dropout(dropout_rate))

    model = keras.models.Sequential(l)
    model.name = model_name
    # model.summary()
    return model
import pickle
def train(model: keras.Model, x, y, learning_rates=(1e-3,)):
    h = pd.DataFrame()
    for lr in learning_rates:
        m = keras.models.clone_model(model)
        opt = keras.optimizers.Adam(learning_rate=lr)
        m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = m.fit(x, y, epochs=20, validation_split=0.2, verbose=0)
        h[f'loss LR={lr:.0e}'] = history.history['loss']
        h[f'val_loss LR={lr:.0e}'] = history.history['val_loss']
        h[f'accuracy LR={lr:.0e}'] = history.history['accuracy']
        h[f'val_accuracy LR={lr:.0e}'] = history.history['val_accuracy']
    h.to_csv(history_dir / f'{model.name}.history.csv')
    return h

def test(model: keras.Model, x, y):
    results = model.evaluate(x, y)
    df = DataFrame(columns=('loss', 'accuracy'), data=[results])
    df.to_csv(history_dir / f'{model.name}.test.csv')
    return df
    

