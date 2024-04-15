import os
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data(data_file: str) -> pd.DataFrame:

    # Define column names based on the provided attributes
    column_names = [
        "subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "PPE"
    ]

    # Load the data into a Pandas DataFrame
    df = pd.read_csv(data_file, names=column_names, header=0)

    return df



def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Preprocess the Parkinson's dataset DataFrame.
    """
    # Remove any unnecessary columns
    df.drop(columns=["subject#"], inplace=True)  # Assuming 'subject#' is not needed for modeling
     # Extract Features (all columns except 'motor_UPDRS' and 'total_UPDRS')
    features = df.drop(columns=['motor_UPDRS', 'total_UPDRS'])
        # Extract Labels ('motor_UPDRS' and 'total_UPDRS' columns)
    labels = df[['motor_UPDRS', 'total_UPDRS']]

    # Calculate and save mean and std for normalization
    mean_std = {}
    for col in df.columns:
        if col != "sex":  # Exclude 'sex' from normalization
            mean = df[col].mean()
            std = df[col].std()
            mean_std[col] = {"mean": mean, "std": std}
            df[col] = (df[col] - mean) / std    # Normalize the data

    # Convert DataFrame to numpy array
    arr = df.to_numpy(dtype=np.float32)
    features_arr = features.to_numpy(dtype=np.float32)
    labels_arr = labels.to_numpy(dtype=np.float32)
    # np.savez("parkinson/features.npz", features)
    # np.savez("parkinson/labels.npz", labels)

    return arr, mean_std, features_arr, labels_arr



def raw_data_plot(df: pd.DataFrame):
    """
    Plot the raw data from the Parkinson's dataset. takes a Pandas dataframe 
    """
    sample_data = df.iloc[1:, 1:10]
    # Create a scatter plot for each column in the DataFrame
    plt.figure(figsize=(10, 6))  # Set the figure size
    # Iterate over each column in the DataFrame (excluding non-numeric columns)
    for i, column in enumerate(sample_data.columns[:-1]):
    # Define a unique color for each column
        color = plt.cm.get_cmap('tab10')(i / len(sample_data.columns[:-1]))  # Use tab10 colormap for colors
        # Plot the data points for the current column
        plt.scatter(sample_data.index, sample_data[column], color=color, label=column)
    # Add labels and title to the plot
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.title('Raw Data')
    plt.grid(True)
    plt.legend()
    plt.show()

def normalized_array_plot(arr: np.ndarray):
    """
    Plot the normalized array from the Parkinson's dataset. takes a numpy array
    """
    sample_data = arr[:, :6]
    # Create a scatter plot for each column in the DataFrame
    plt.figure(figsize=(10, 6))  # Set the figure size
    # Iterate over each column in the DataFrame (excluding non-numeric columns)
    for i in range(sample_data.shape[1]):
    # Define a unique color for each column
        color = plt.cm.get_cmap('tab10')(i / sample_data.shape[1])  # Use tab10 colormap for colors
        # Plot the data points for the current column
        plt.scatter(sample_data[:, i], range(sample_data.shape[0]), color=color, label=f"Column {i}")
    # Add labels and title to the plot
    plt.ylabel('Data Point Index')
    plt.xlabel('Value')
    plt.title('Normalized Data')
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    if not os.path.exists("parkinson"):
        os.makedirs("parkinson/a", exist_ok=True)
        os.makedirs("parkinson/b", exist_ok=True)
        os.makedirs("parkinson/c", exist_ok=True)

    df = load_data("parkinsons_updrs.data")

    # Display the first few rows of the DataFrame to inspect the loaded data
    print(df.head())
    
    raw_data_plot(df)


    # Preprocess the dataset
    arr, mean_std, features, labels = preprocess_data(df)
    print("shape of arr", arr.shape)
  
    #normalized_array_plot(arr)
    

    # data, label = split_feature(arr, mean_std, "parkinson")
    print("shape of data", features.shape)
    # Split the data into 70%training and 30%testing sets
    #####Do I want labels in train and test set?##########
    train, test, label_train, label_test = train_test_split(arr, labels, test_size=0.3, random_state=42)
    np.savez("parkinson/parkinson.train.npz", train)
    np.savez("parkinson/parkinson.test.npz", test)
    np.savez("parkinson/labels_train.npz", label_train)
    np.savez("parkinson/labels_test.npz", label_test)
    # a = np.concatenate((train, label_train), axis=1)
    # b = np.concatenate((test, label_test), axis=1)
    #split the train futher into a and b
    print("shape of train", train.shape)
    a_train = train[:, :11]
    b_train = train[:, 11:]

    print("shape of a_train", a_train.shape)
    print("shape of b_train", b_train.shape)

    print("shape of test", test.shape)
    #split test further into a and b
    a_test = test[:, :11]
    b_test = test[:, 11:]

    print("shape of a_test", a_test.shape)
    print("shape of b_test", b_test.shape)

    print("shape of label_train", label_train.shape)
    print("shape of label_test", label_test.shape)


    
    # Ensure the directory exists before saving the file
    if not os.path.exists("parkinson/a"):
        os.makedirs("parkinson/a", exist_ok=True)
    if not os.path.exists("parkinson/b"):
        os.makedirs("parkinson/b", exist_ok=True)
    if not os.path.exists("parkinson/c"):
        os.makedirs("parkinson/c", exist_ok=True)
    
    np.savez("parkinson/a/train.npz", a_train)
    np.savez("parkinson/b/train.npz", b_train)
    np.savez("parkinson/c/train.npz", label_train)


    
    # Ensure the directory exists before saving the file
    if not os.path.exists("parkinson/a"):
        os.makedirs("parkinson/a", exist_ok=True)
    np.savez("parkinson/a/test.npz", a_test)
    np.savez("parkinson/b/test.npz", b_test)
    np.savez("parkinson/c/test.npz", label_test)