import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt 
from model import MLP
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def load_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    tensor = torch.tensor(arr)
    return tensor


def make_dataloader(filename: str, datafile: str, batch_size: int = None, shuffle: bool = False,
                    drop_last: bool = False) -> DataLoader:
    tensor = load_tensor(filename)
    feature= tensor[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    feature = feature.float()
    # Extract labels (columns 3 and 4)
    label = tensor[:, [3, 4]]  # Selecting columns 3 and 4 as labels
    label = label.float()
    dataset = TensorDataset(feature, label)
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def train(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module, optimizer: optim.Optimizer):
    model.train()
    count = len(dataloader)
    total_loss = 0
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()
    return total_loss / count


def validate(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module):
    model.eval()

    count = len(dataloader)
    total_loss = 0
    pred_probs = []
    true_ys = []
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, file=sys.stdout):
            out = model(xs)
            # Accumulate predictions and labels
            pred_probs.append(out)  # Collect model predictions
            true_ys.append(ys)       # Collect ground truth labels
            loss_val = loss(out, ys)
            total_loss += loss_val.item()

    # Post-processing: Flatten and calculate metrics
    pred_probs = torch.cat(pred_probs, dim=0)  # Concatenate all predictions
    true_ys = torch.cat(true_ys, dim=0)         # Concatenate all labels
    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse = mean_squared_error(true_ys.numpy(), pred_probs.numpy())
    mae = mean_absolute_error(true_ys.numpy(), pred_probs.numpy())
    r2 = r2_score(true_ys.numpy(), pred_probs.numpy())

    return total_loss / count, mse, mae, r2





if __name__ == '__main__':
    train_filename = "parkinson/parkinson.train.npz" #train and test only include features
    test_filename = "parkinson/parkinson.test.npz"
    epochs = 150
    batch_size = 22
    lr = 1e-4
    eval_every = 1
    # List to store training losses
    training_losses = []  
    val_losses = []
    all_mse = []
    all_mae = []
    all_r2 = []
     
    train_dataloader = make_dataloader(train_filename, "train", batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = make_dataloader(test_filename, "test", batch_size=batch_size, shuffle=False, drop_last=False)

    mlp = MLP(input_size=17)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr)

    for epoch in range(epochs):
        train_loss = train(train_dataloader, mlp, loss, optimizer)
        print(f"epoch: {epoch}, train loss: {train_loss}")
        training_losses.append(train_loss)
        if epoch % eval_every == 0:
            validate_loss, mse, mae, r2= validate(test_dataloader, mlp, loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, MSE: {mse}, MAE: {mae}")
            val_losses.append(validate_loss)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)
  
    print(f"Training Losses: {training_losses}")
    print(f"Validation Losses: {val_losses}")

    # Plot the training loss as a function of epoch
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (MSE) vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    #plot mse, mae and r2 as function of epoch
    plt.plot(range(1, epochs+1), all_mse, label='MSE', color='red')
    plt.plot(range(1, epochs+1), all_mae, label='MAE', color='blue')
    plt.plot(range(1, epochs+1), all_r2, label='R2', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Metrics vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    final_loss, mse, mae, r2 = validate(test_dataloader, mlp, loss)
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
    




            
            