from os import path

import torch
from utils import make_dataloaders
import glob
from utils import create_loss_meters, update_losses, log_results, visualize
from tqdm import tqdm
from mainmodel import MainModel
import numpy as np


DATA_PATH = 'data'
BATCH_SIZE = 16


def train_model(model, train_dl, epochs, display_every=200):
    # data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        # function returing a dictionary of objects to
        loss_meter_dict = create_loss_meters()
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            # function updating the log objects
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                # function to print out the losses
                log_results(loss_meter_dict)
                # visualize(model, data, save=False) # function displaying the model's outputs


def main():
    paths = glob.glob(DATA_PATH + "/*.jpg")
    # print(paths)
    # np.random.seed(123)
    paths_subset = np.random.choice(paths, len(paths), replace=False)
    rand_idxs = np.random.permutation(len(paths))
    train_idxs = rand_idxs[:int(0.8*len(rand_idxs))]
    val_idxs = rand_idxs[int(0.8*len(rand_idxs)):]
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    # print(len(train_paths), len(val_paths))
    train_dl = make_dataloaders(
        paths=train_paths, batch_size=32, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    model = MainModel()
    train_model(model, train_dl, 100)


if __name__ == '__main__':
    main()
