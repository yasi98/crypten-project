#####   Train encrypted MLP model on 3 workers  ########

import sys
import crypten
import crypten.communicator as comm
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from data_utils import crypten_collate
from model import MLP
from collections import defaultdict, OrderedDict



###### Globals ######

# Define dataset names and corresponding feature sizes
names = ["a", "b", "c"]
feature_sizes = [11, 10, 2]

#   Will use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


########################


#   Function to load local tensor data
def load_local_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    arr = np.copy(arr)  # Make a writable copy of the array
    tensor = torch.tensor(arr, dtype=torch.float32).to(device)
    return tensor

#   Function to load encrypted tensor data
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
            dummy_tensor = torch.zeros((count, feature_size), dtype=torch.float32).to(device)
            tensor = crypten.cryptensor(dummy_tensor, src=i)
        encrypt_tensors.append(tensor)

    res = crypten.cat(encrypt_tensors, dim=1)
    return res

#   Function to create a local dataloader
def make_local_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    tensor = load_local_tensor(filename)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

#   Function to create an encrypted MPC model
def make_mpc_model(local_model: torch.nn.Module, sample_input: torch.Tensor):
    dummy_input = torch.empty((1, 17)).to(device)
    model = crypten.nn.from_pytorch(local_model, dummy_input)
    model.encrypt() #encrypt the model with crypten
    return model


#   Function to create an encrypted dataloader
def make_mpc_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    
    mpc_tensor = load_encrypt_tensor(filename)
    feature= mpc_tensor[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].to(device)
    #uses a custom collation function for combining individual data samples into batches and a specified random number generator for shuffling the data.
    label = mpc_tensor[:, [3, 4]].to(device)  # Selecting columns 3 and 4 as labels
    dataset = TensorDataset(feature, label)
    # Generate a random seed for the dataloader
    seed = (crypten.mpc.MPCTensor.rand(1) * (2 ** 32)).get_plain_text().int().item()
    generator = torch.Generator()
    generator.manual_seed(seed)
    # Create the dataloader with the crypten_collate function
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=crypten_collate, generator=generator)
    return dataloader


#   Function to train the encrypted model
def train_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, lr: float):
    
    total_loss = None
    count = len(dataloader)
    gradients = []
    model.train()

    for xs, ys in tqdm(dataloader, file=sys.stdout):
        
        xs.requires_grad = True
        out = model(xs)
        loss_val = loss(out, ys)

        model.zero_grad()
   
        loss_val.backward()
        #optimizer.step()
        gradients.append(xs.grad.clone())
        #gradient_dict[name] = param.grad.clone().detach()
        model.update_parameters(lr)


        # create a new tensor that doesn't require gradients
        # detach loss_val from the computation graph
        # This ensures that subsequent operations won't affect the computation of gradients for total_loss.
        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()
    if gradients:
        #   storing gradients to be used for model explainability
        gradients_tensor = crypten.cat(gradients, dim=0)
        abs_gradients = torch.abs(gradients_tensor.get_plain_text())
        mean_gradients = abs_gradients.mean(dim=0)

    total_loss = total_loss.get_plain_text().item()

    return total_loss / count, mean_gradients

#   Function to validate the encrypted model
def validate(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module):

    model.eval()
    pred_ys = []
    true_ys = []
    total_loss = None
    count = len(dataloader)

    for xs, ys in tqdm(dataloader, file=sys.stdout):
        
        out = model(xs)
        loss_val = loss(out, ys)
        loss_val.backward()  # Compute gradients
    
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
    
    # Initialize Crypten
    crypten.init()
    crypten.common.serial.register_safe_class(MLP)

    #   Load datasets, previously saved by data_preprocessing.py script
    rank = comm.get().get_rank()
    name = names[rank]
    train_filename = f"parkinson/{name}/train.npz"
    test_filename = f"parkinson/{name}/test.npz"

    train_dataloader = make_mpc_dataloader(train_filename, batch_size, shuffle=True, drop_last=False)
    test_dataloader = make_mpc_dataloader(test_filename, batch_size, shuffle=False, drop_last=False)
    # Create a sample input tensor from data
    sample_data = load_local_tensor(test_filename)
    sample_input = sample_data[:1, :]  # Use the first row as a sample input
    # print("sample input size", sample_input.shape)
    # input_size = sample_input.shape[1] 
    
    # Define model and loss function
    model = MLP(input_size=17).to(device)
    #optimizer = crypten.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    mpc_model = make_mpc_model(model, sample_input=sample_input)
    mpc_loss = crypten.nn.MSELoss()
    

    ########## Start of training loop ###########

    for epoch in range(epochs):
    #for batch_X, batch_y in train_dataloader:
        train_loss, gradients = train_mpc(train_dataloader, mpc_model, mpc_loss, lr)
        print(f"epoch: {epoch}, train loss: {train_loss}")
        training_losses.append(train_loss)
        all_grads.append(gradients)

        if epoch % eval_every == 0:
            validate_loss, mse, mae, r2= validate(test_dataloader, mpc_model, mpc_loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, MSE: {mse}, MAE: {mae}")
            val_losses.append(validate_loss)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)


   #######################
 
    final_loss, mse, mae, r2 = validate(test_dataloader, mpc_model, mpc_loss)
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
    all_grads.append(gradients)
    if all_grads:
        all_grads = torch.stack(all_grads)
        feature_importance_scores = torch.abs(all_grads).mean(dim=0)
    feature_importance_pairs = list(zip(feature_names, feature_importance_scores))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 Features:")
    for i, (name, score) in enumerate(feature_importance_pairs[:5]):
        print(f"{i+1}. {name}: {score.item()}")




    # Create a LIME explainer
    def predict_fn(x, model: crypten.nn.Module):
        model = model.decrypt()
        # Convert input data to a tensor and perform necessary preprocessing
        input_tensor = torch.tensor(x, dtype=torch.float32)
        print("shape of input", input_tensor.shape)
    # Use the decrypted model to make predictions
        with crypten.no_grad():
            predictions = model(input_tensor)
            # Convert encrypted predictions to plaintext
            predictions = predictions.numpy()

        return predictions


    train = train_dataloader.dataset.tensors[0]
    train = train.get_plain_text().numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=['motor_UPDRSA','total_UPDRS'], verbose=True, mode='regression')
    # Select an instance from the test dataset for explanation (e.g., index 25)
    instance = train[25]
    #print("shape of instance", instance.shape)
    exp = explainer.explain_instance(instance, lambda x: predict_fn(x, mpc_model), num_features=len(feature_names))
    #exp.show_in_notebook(show_table=True)
    exp.save_to_file('lime_explanation.html')
    
    crypten.save(mpc_model, "model.pth")


########### End of main  ############




        
#   Call the main function
if __name__ == '__main__':
    main()
