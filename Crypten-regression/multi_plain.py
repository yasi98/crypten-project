#####   Train encrypted MLP model on 3 workers  ########

import sys
import os
import numpy as np
import torch
import crypten.communicator as comm
import crypten
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MLP
from collections import defaultdict, OrderedDict
import torch.distributed as dist



###### Globals ######

# Define dataset names and corresponding feature sizes
names = ["a", "b", "c"]
feature_sizes = [11, 10, 2]

#   Will use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#dist.init_process_group(backend='gloo')
#dist.barrier()

########################


#   Function to load local tensor data
def load_local_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    arr = np.copy(arr)  # Make a writable copy of the array
    tensor = torch.tensor(arr, dtype=torch.float32).to(device)
    return tensor

def load_encrypt_tensor(filename: str) -> crypten.CrypTensor:
    local_tensor = load_local_tensor(filename)
    rank = comm.get().get_rank()
    #rank = int(os.environ.get('RANK', '0')) 
    count = local_tensor.shape[0]

    # Define a list to store tensors
    tensors = []

    # Iterate over names and feature_sizes
    for i, (name, feature_size) in enumerate(zip(names, feature_sizes)):
        if rank == i:
            assert local_tensor.shape[1] == feature_size, \
            f"{name} feature size should be {feature_size}, but got {local_tensor.shape[1]}"
            tensor = torch.tensor(local_tensor, dtype=torch.float32).to(device)  # Create PyTorch tensor
        else:
            dummy_tensor = torch.zeros((count, feature_size), dtype=torch.float32).to(device)
            tensor = dummy_tensor  # Use dummy tensor
        tensors.append(tensor)

    # Concatenate tensors along the specified dimension
    res = torch.cat(tensors, dim=1)
    return res

#   Function to create a local dataloader
def make_local_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    tensor = load_local_tensor(filename)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader



#   Function to create an dataloader
def make_mpc_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    
    tensor = load_encrypt_tensor(filename)
    feature= tensor[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].to(device)
    #uses a custom collation function for combining individual data samples into batches and a specified random number generator for shuffling the data.
    label = tensor[:, [3, 4]].to(device)  # Selecting columns 3 and 4 as labels
    dataset = TensorDataset(feature, label)
    # Generate a random seed for the dataloader
    seed = torch.randint(0, 2**32, (1,))
    generator = torch.Generator()
    generator.manual_seed(seed.item())

    
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator)
    return dataloader


#   Function to train the  model
def train_model(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module, lr: float, optimizer: optim.Optimizer):
    
    total_loss = None
    count = len(dataloader)
    gradients = []
    model.train()
    

    for xs, ys in tqdm(dataloader, file=sys.stdout):
        
        xs.requires_grad = True
        out = model(xs)
        loss_val = loss(out, ys)

        optimizer.zero_grad()
   
        loss_val.backward()
        #optimizer.step()
        gradients.append(xs.grad.clone())
        #gradient_dict[name] = param.grad.clone().detach()
        optimizer.step()


        # create a new tensor that doesn't require gradients
        # detach loss_val from the computation graph
        # This ensures that subsequent operations won't affect the computation of gradients for total_loss.
        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()
    """  if gradients:
        #   storing gradients to be used for model explainability
        gradients_tensor = torch.cat(gradients, dim=0)
        abs_gradients = torch.abs(gradients_tensor)
        mean_gradients = abs_gradients.mean(dim=0) """

    total_loss = total_loss.item()

    return total_loss / count

#   Function to validate the  model
def validate(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module):
    
    model.eval()
    pred_ys = []
    true_ys = []
    total_loss = 0.0
    count = len(dataloader)

    with torch.no_grad():  # Disable gradient tracking during validation
        for xs, ys in tqdm(dataloader, file=sys.stdout):
            out = model(xs)
            loss_val = loss(out, ys)
            total_loss += loss_val.item()

            pred_ys.append(out)
            true_ys.append(ys)

    # Concatenate predictions and true labels
    pred_ys = torch.cat(pred_ys, dim=0)
    true_ys = torch.cat(true_ys, dim=0)

    # Convert predictions and true labels to numpy arrays
    pred_ys = pred_ys.cpu().numpy()
    true_ys = true_ys.cpu().numpy()

    # Calculate evaluation metrics
    mse = mean_squared_error(true_ys, pred_ys)
    mae = mean_absolute_error(true_ys, pred_ys)
    r2 = r2_score(true_ys, pred_ys)
    
    return total_loss / count, mse, mae, r2


########   Main  #############

def main():

    #   Hyperparameters
    epochs = 150
    batch_size = 22
    lr = 1e-3
    eval_every = 1

    #   List to store performance metrics
    training_losses = []  
    all_grads = []
    val_losses = []
    all_mse = []
    all_mae = []
    all_r2 = []
    

      # Define model and loss function
    model = MLP(input_size=17, ).to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss = torch.nn.MSELoss()

    #   Load datasets, previously saved by data_preprocessing.py script
    
   # Get the rank of the current process
    crypten.init()
    rank = comm.get().get_rank()
    #rank = int(os.environ.get('RANK', '0')) 
    name = names[rank]
    train_filename = os.path.join(f"parkinson/{name}/train.npz")
    test_filename = os.path.join(f"parkinson/{name}/test.npz")

    train_dataloader = make_mpc_dataloader(train_filename, batch_size, shuffle=True, drop_last=False)
    test_dataloader = make_mpc_dataloader(test_filename, batch_size, shuffle=False, drop_last=False)
    # Create a sample input tensor from data
    sample_data = load_local_tensor(test_filename)
    sample_input = sample_data[:1, :]  # Use the first row as a sample input
    # print("sample input size", sample_input.shape)
    # input_size = sample_input.shape[1] 
    
  
    #optimizer = crypten.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    

    ########## Start of training loop ###########

    for epoch in range(epochs):
    #for batch_X, batch_y in train_dataloader:
        train_loss = train_model(train_dataloader, model, loss, lr, optimizer)
        print(f"epoch: {epoch}, train loss: {train_loss}")
        training_losses.append(train_loss)
        #all_grads.append(gradients)

        if epoch % eval_every == 0:
            validate_loss, mse, mae, r2= validate(test_dataloader, model, loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, MSE: {mse}, MAE: {mae}")
            val_losses.append(validate_loss)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)


   #######################
 
    final_loss, mse, mae, r2 = validate(test_dataloader, model, loss)
    #feature_importance_scores = torch.abs(gradients.get_plain_text()).mean(dim=0)  # Mean absolute value of gradients
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")

    #   Following is to be used for drawing plots later
    print('training loss', training_losses)
    print('validation loss', val_losses)
    print('mse', all_mse)
    print('mae', all_mae)
    print('r2', all_r2)

    #   Plot the training loss as a function of epoch
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (MSE) vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    #   Plot mse, mae and r2 as function of epoch
    plt.plot(range(1, epochs+1), all_mse, label='MSE', color='red')
    plt.plot(range(1, epochs+1), all_mae, label='MAE', color='blue')
    plt.plot(range(1, epochs+1), all_r2, label='R2', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Metrics vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()


  
    feature_names = [
        "age", "sex", "test_time",
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "PPE"
    ]
"""     all_grads.append(gradients)
    if all_grads:
        all_grads = torch.stack(all_grads)
        feature_importance_scores = torch.abs(all_grads).mean(dim=0)
    feature_importance_pairs = list(zip(feature_names, feature_importance_scores))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 Features:")
    for i, (name, score) in enumerate(feature_importance_pairs[:5]):
        print(f"{i+1}. {name}: {score.item()}")
 """



    #model.save(mpc_model, "model.pth")


########### End of main  ############




        
#   Call the main function
if __name__ == '__main__':
    main()
