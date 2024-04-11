import sys

import crypten
import crypten.communicator as comm
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import crypten_collate
from model import MLP

# Define dataset names and corresponding feature sizes
names = ["a", "b", "c"]
feature_sizes = [11, 10, 2]

# Function to load local tensor data
def load_local_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    arr = np.copy(arr)  # Make a writable copy of the array
    tensor = torch.tensor(arr, dtype=torch.float32)
    return tensor

# Function to load encrypted tensor data
def load_encrypt_tensor(filename: str) -> crypten.CrypTensor:
    local_tensor = load_local_tensor(filename)
    rank = comm.get().get_rank()
    count = local_tensor.shape[0]

    encrypt_tensors = []
    for i, (name, feature_size) in enumerate(zip(names, feature_sizes)):
        if rank == i:
            assert local_tensor.shape[1] == feature_size, \
                f"{name} feature size should be {feature_size}, but get {local_tensor.shape[1]}"
            tensor = crypten.cryptensor(local_tensor, src=i)
        else:
            dummy_tensor = torch.zeros((count, feature_size), dtype=torch.float32)
            tensor = crypten.cryptensor(dummy_tensor, src=i)
        encrypt_tensors.append(tensor)

    res = crypten.cat(encrypt_tensors, dim=1)
    return res

# Function to create a local dataloader
def make_local_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    tensor = load_local_tensor(filename)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

# Function to create an MPC model
def make_mpc_model(local_model: torch.nn.Module, sample_input: torch.Tensor):
    dummy_input = torch.empty((1, 17))
    model = crypten.nn.from_pytorch(local_model, dummy_input)
    model.encrypt() #encrypt the model with crypten
    return model


def make_mpc_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    
    mpc_tensor = load_encrypt_tensor(filename)
    feature= mpc_tensor[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    #uses a custom collation function for combining individual data samples into batches and a specified random number generator for shuffling the data.
    label = mpc_tensor[:, [3, 4]]
    dataset = TensorDataset(feature, label)
    # Generate a random seed for the dataloader
    seed = (crypten.mpc.MPCTensor.rand(1) * (2 ** 32)).get_plain_text().int().item()
    generator = torch.Generator()
    generator.manual_seed(seed)
    # Create the dataloader with the crypten_collate function
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=crypten_collate, generator=generator)
    return dataloader


# Function to train the MPC model
def train_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, lr: float):
    total_loss = None
    count = len(dataloader)

    model.train()
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        model.zero_grad()
        loss_val.backward()
        model.update_parameters(lr)


        # create a new tensor that doesn't require gradients
        # detach loss_val from the computation graph
        # This ensures that subsequent operations won't affect the computation of gradients for total_loss.
        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()
    return total_loss / count

# Function to validate the MPC model
def validate(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module):
    model.eval()
    pred_ys = []
    true_ys = []
    total_loss = None
    count = len(dataloader)
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        pred_ys.append(out)
        true_ys.append(ys)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()
    # for item in pred_ys:
    #     print("type of pred", type(item))
    # Flatten and concatenate predictions and true labels for evaluation
    pred_ys = crypten.cat(pred_ys, dim=0)
    true_ys = crypten.cat(true_ys, dim=0)
    # Convert predictions and true labels to plain text for evaluation
    pred_ys = pred_ys.get_plain_text()
    true_ys = true_ys.get_plain_text()
    # Convert predictions and true labels to Python lists
    pred_ys = pred_ys.tolist()
    true_ys = true_ys.tolist()

    # Calculate eval metrics
    mse = mean_squared_error(true_ys, pred_ys)
    mae = mean_absolute_error(true_ys, pred_ys)
    r2 = r2_score(true_ys, pred_ys)
    
    return total_loss / count, mse, mae, r2


def test():
    crypten.init()
    filename = "parkinson/parkinson.test.npz"

    mpc_tensor = load_encrypt_tensor(filename)
    feature, label = mpc_tensor[:32, :-1], mpc_tensor[:32, -1]
    print(feature.shape, feature.ptype)

    model = MLP()
    mpc_model = make_mpc_model(model)
    loss = crypten.nn.MSELoss()

    mpc_model.train()
    out = mpc_model(feature)
    prob = out.sigmoid()
    loss_val = loss(prob, label)

    mpc_model.zero_grad()
    loss_val.backward()
    mpc_model.update_parameters(1e-3)


def main():
    epochs = 50
    batch_size = 32
    lr = 1e-3
    eval_every = 1
    # List to store training losses
    training_losses = []  
    crypten.init()

    rank = comm.get().get_rank()
    name = names[rank]
    train_filename = f"parkinson/{name}/train.npz"
    test_filename = f"parkinson/{name}/test.npz"

    train_dataloader = make_mpc_dataloader(train_filename, batch_size, shuffle=True, drop_last=False)
    test_dataloader = make_mpc_dataloader(test_filename, batch_size, shuffle=False, drop_last=False)
    # Create a sample input tensor from your data
    sample_data = load_local_tensor(test_filename)
    sample_input = sample_data[:1, :]  # Use the first row as a sample input
    # print("sample input size", sample_input.shape)
    # input_size = sample_input.shape[1] 
    
    model = MLP(input_size=17)
    mpc_model = make_mpc_model(model, sample_input=sample_input)
    mpc_loss = crypten.nn.MSELoss()

    for epoch in range(epochs):
    #for batch_X, batch_y in train_dataloader:
        train_loss = train_mpc(train_dataloader, mpc_model, mpc_loss, lr)
        print(f"epoch: {epoch}, train loss: {train_loss}")
        training_losses.append(train_loss)

        if epoch % eval_every == 0:
            validate_loss, mse, mae, r2= validate(test_dataloader, mpc_model, mpc_loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, MSE: {mse}, MAE: {mae}")
 
    final_loss, mse, mae, r2 = validate(test_dataloader, mpc_model, mpc_loss)
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
    print('training loss', training_losses)
    #   Plot the training loss as a function of epoch
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (MSE) vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

  
    
     




            


if __name__ == '__main__':
    main()
