import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import AdaptiveAvgPool2d
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


def MNIST_tensorflow_PCA(class_set, train_dataset_number, test_dataset_number, batch_size, nbr_rows, nbr_cols):
    """ Load the MNIST dataset from tensorflow and apply PCA to reduce the dimensionality.
    Args:
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - batch_size: size of the batch (int)
        - nbr_rows: number of rows in the image (int)
        - nbr_cols: number of columns in the image (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
        """
    nbr_components = nbr_rows*nbr_cols # Number of components to keep after PCA

    x_train, y_train, x_test, y_test = Load_MNIST_tensorflow()
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array = [], []
    test_list_data_array, test_list_label_array = [], []
    for i in range(x_train.shape[0]):
        if (y_train[i] in class_set) and (len(train_list_data_array) < train_dataset_number):
            train_list_data_array.append(x_train[i].reshape(28*28))
            train_list_label_array.append(int(y_train[i]))
        if (len(train_list_data_array) == train_dataset_number):
            break
    for i in range(x_test.shape[0]):
        if (y_test[i] in class_set) and (len(test_list_data_array) < test_dataset_number):
            test_list_data_array.append(x_test[i].reshape(28*28))
            test_list_label_array.append(int(y_test[i]))
        if (len(test_list_data_array) == test_dataset_number):
            break
        
    samples = np.concatenate((np.array(train_list_data_array), np.array(test_list_data_array)), axis=0)

    pca = PCA(nbr_components) 
    samples_pca = pca.fit_transform(samples)
    x_train, x_test = samples_pca[:train_dataset_number], samples_pca[train_dataset_number:]
    
    # Convert numpy arrays to PyTorch tensors
    tensor_train_data_pca, tensor_test_data_pca = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
    tensor_train_label , tensor_test_label = torch.tensor(test_list_label_array, dtype=torch.long), torch.tensor(train_list_label_array, dtype=torch.long)

    print("Train data and label tensors of size:{} and {}".format(tensor_train_data_pca.shape, tensor_train_label.shape))
    print("Test data and label tensors of size:{} and {}".format(tensor_test_data_pca.shape, tensor_test_label.shape))

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_train_data_pca, tensor_train_label), batch_size)
    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_test_data_pca, tensor_test_label), batch_size)
    return train_dataloader, test_dataloader




def MNIST_tensorflow_Pooling(class_set, train_dataset_number, test_dataset_number, batch_size, nbr_rows, nbr_cols):
    """ Load the MNIST dataset from tensorflow and apply pooling to reduce the dimensionality.
    Args:
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - batch_size: size of the batch (int)
        - nbr_rows: number of rows in the image (int)
        - nbr_cols: number of columns in the image (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    nbr_components = nbr_rows*nbr_cols # Number of components to keep
    x_train, y_train, x_test, y_test = Load_MNIST_tensorflow()
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array = [], []
    test_list_data_array, test_list_label_array = [], []
    for i in range(x_train.shape[0]):
        if (y_train[i] in class_set) and (len(train_list_data_array) < train_dataset_number):
            train_list_data_array.append(x_train[i].reshape(28*28))
            train_list_label_array.append(int(y_train[i]))
        if (len(train_list_data_array) == train_dataset_number):
            break
    for i in range(x_test.shape[0]):
        if (y_test[i] in class_set) and (len(test_list_data_array) < test_dataset_number):
            test_list_data_array.append(x_test[i].reshape(28*28))
            test_list_label_array.append(int(y_test[i]))
        if (len(test_list_data_array) == test_dataset_number):
            break  
    samples = np.concatenate((np.array(train_list_data_array), np.array(test_list_data_array)), axis=0)
    samples_pooling = []
    Pooling = AdaptiveAvgPool2d((nbr_rows, nbr_cols))
    for image in samples:
        image = image.reshape(28,28)
        image = torch.tensor(image).unsqueeze(0).float()
        image = Pooling(image)
        samples_pooling.append(image.squeeze(0))
    x_train, x_test = samples_pooling[:train_dataset_number], samples_pooling[train_dataset_number:]
    x_train, x_test = np.array(x_train), np.array(x_test)
    
    # Convert numpy arrays to PyTorch tensors
    tensor_train_data_pca, tensor_test_data_pca = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
    tensor_train_label , tensor_test_label = torch.tensor(test_list_label_array, dtype=torch.long), torch.tensor(train_list_label_array, dtype=torch.long)

    print("Train data and label tensors of size:{} and {}".format(tensor_train_data_pca.shape, tensor_train_label.shape))
    print("Test data and label tensors of size:{} and {}".format(tensor_test_data_pca.shape, tensor_test_label.shape))

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_train_data_pca, tensor_train_label), batch_size)
    test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_test_data_pca, tensor_test_label), batch_size)
    return train_dataloader, test_dataloader

def Digits_sklearn_Pooling(class_set, train_dataset_number, test_dataset_number, batch_size, nbr_rows, nbr_cols):
    """ Load the MNIST dataset from tensorflow and apply pooling to reduce the dimensionality.
    Args:
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - batch_size: size of the batch (int)
        - nbr_rows: number of rows in the image (int)
        - nbr_cols: number of columns in the image (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    nbr_components = nbr_rows*nbr_cols # Number of components to keep
    (x_train, y_train), (x_test, y_test) = Load_Digits_sklearn()