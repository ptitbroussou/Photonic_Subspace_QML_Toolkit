import numpy as np
import torch
from tensorflow.keras.datasets import mnist
from sklearn import datasets
from sklearn.decomposition import PCA


# Load the MNIST digits dataset
def Load_MNIST_tensorflow():
    """ Load the MNIST dataset from tensorflow. 
    Output:
        - x_train: training data of size (60000, 28, 28)
        - y_train: training labels of size (60000,)
        - x_test: testing data of size (10000, 28, 28)
        - y_test: testing labels of size (10000,)
        """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


# Load the digits dataset:
def Load_Digits_sklearn():
    """ Load the digits dataset from sklearn. Each image is of size 8x8 pixels.
    Output:
        - x_train: training data of size (750, 64)
        - y_train: training labels of size (750,)
        - x_test: testing data of size (1047, 64)
        - y_test: testing labels of size (1047,)
        """
    digits = datasets.load_digits()
    (x_train, y_train), (x_test, y_test) = (digits.data[:750], digits.target[:750]), (digits.data[750:], digits.target[750:])
    return (x_train, y_train), (x_test, y_test)


def select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set):
    """ Select only the classes in class_set.
    Args:
        - x_train: training data
        - y_train: training labels
        - x_test: testing data
        - y_test: testing labels
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
    Output:
        - train_list_data_array: training data as a list of numpy arrays
        - test_list_data_array: testing data as a list of numpy arrays
        - train_list_label_array: training labels as a list of numpy arrays
        - test_list_label_array: testing labels as a list of numpy arrays
    """
    train_list_data_array, train_list_label_array = [], []
    test_list_data_array, test_list_label_array = [], []
    for i in range(x_train.shape[0]):
        if (y_train[i] in class_set) and (len(train_list_data_array) < train_dataset_number):
            train_list_data_array.append(x_train[i])
            train_list_label_array.append(int(y_train[i]))
    for i in range(x_test.shape[0]) :
        if (y_test[i] in class_set) and (len(test_list_data_array) < test_dataset_number):
            test_list_data_array.append(x_test[i])
            test_list_label_array.append(int(y_test[i]))
    return train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array

def create_pytorch_dataset(train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array, batch_size):
    """ Create PyTorch tensors from the input data.
    Args:
        - train_list_data_array: training data as a list of numpy arrays
        - test_list_data_array: testing data as a list of numpy arrays
        - train_list_label_array: training labels as a list of numpy arrays
        - test_list_label_array: testing labels as a list of numpy arrays
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
     # Convert numpy arrays to PyTorch tensors
    tensor_train_data, tensor_test_data = torch.tensor(train_list_data_array, dtype=torch.float32), torch.tensor(test_list_data_array, dtype=torch.float32)
    tensor_train_label , tensor_test_label = torch.tensor(train_list_label_array, dtype=torch.long), torch.tensor(test_list_label_array, dtype=torch.long)
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_train_data, tensor_train_label), batch_size)
    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_test_data, tensor_test_label), batch_size)
    return train_dataloader, test_dataloader


def PCA_reduction(train_list_data_array, test_list_data_array, train_dataset_number, nbr_components):
    """ Perform PCA reduction on the dataset.
    Args:
        - train_list_data_array: training data as a list of numpy arrays
        - test_list_data_array: testing data as a list of numpy arrays
        - train_dataset_number: number of training samples
        - nbr_components: number of components to keep after PCA
    Output:
        - x_train: training data after PCA
        - x_test: testing data after PCA
    """
    samples = np.concatenate((np.array(train_list_data_array), np.array(test_list_data_array)), axis=0)

    pca = PCA(nbr_components) 
    samples_pca = pca.fit_transform(samples)
    x_train, x_test = samples_pca[:train_dataset_number], samples_pca[train_dataset_number:]
    return x_train, x_test

