import numpy as np
import matplotlib.pyplot as plt
import torch
import pennylane as qml
from torch.nn import AdaptiveAvgPool2d
from sklearn.decomposition import PCA

#import medmnist
#rom medmnist import INFO

from Data_Loading.toolbox_data import *
 

##################################################################################################################
# Raw Data
##################################################################################################################
def MNIST_tensorflow_raw(class_set, train_dataset_number, test_dataset_number, batch_size):
    """ Load the MNIST dataset from tensorflow.
    Args:
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    (x_train, y_train), (x_test, y_test) = Load_MNIST_tensorflow()
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    print("Train data and label tensors of size:{} and {}".format(np.shape(train_list_data_array), np.shape(train_list_label_array)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(test_list_data_array), np.shape(test_list_label_array)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array, batch_size)

def Digits_sklearn_raw(class_set, train_dataset_number, test_dataset_number, batch_size):
    """ Load the digits dataset from sklearn.
    Args:
        - class_set: set of classes to keep as a list (example: [0, 1, 2, 3])
        - train_dataset_number: number of training samples to keep (int)
        - test_dataset_number: number of testing samples to keep (int)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    (x_train, y_train), (x_test, y_test) = Load_Digits_sklearn()
    x_train, x_test = x_train.reshape(x_train.shape[0], 8,8), x_test.reshape(x_test.shape[0], 8,8)
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    print("Train data and label tensors of size:{} and {}".format(np.shape(train_list_data_array), np.shape(train_list_label_array)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(test_list_data_array), np.shape(test_list_label_array)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array, batch_size)

def BAS_pennylane_raw(size, train_dataset_number, test_dataset_number, batch_size):
    """ Load the BAS dataset from Pennylane. The dataset is composed only composed of 2 classes.
    Args:
        - size: size (4x4, 8x8, 16x16, or 32x32) of the square image (int) 
        - train_dataset_number: number of training samples to keep (int, less than 1000)
        - test_dataset_number: number of testing samples to keep (int, less than 200)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    size_str = str(size)
    [ds] = qml.data.load("other", name="bars-and-stripes")
    x_train = np.array(ds.train[size_str]['inputs']) # vector representations images
    y_train = np.array(ds.train[size_str]['labels']) # labels for the above images  
    x_test = np.array(ds.test[size_str]['inputs']) # vector representations of images
    y_test = np.array(ds.test[size_str]['labels']) # labels for the above images
    # Reshape the data:
    x_train, x_test = x_train[:train_dataset_number].reshape(train_dataset_number, size,size), x_test[:test_dataset_number].reshape(test_dataset_number, size,size)
    y_train, y_test = y_train[:train_dataset_number], y_test[:test_dataset_number]
    # Transform label -1 to label 0:
    y_train[y_train == -1], y_test[y_test == -1] = 0, 0
    print("Train data and label tensors of size:{} and {}".format(np.shape(x_train), np.shape(y_train)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(x_test), np.shape(y_test)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)

def BAS_pennylane_raw_MSE(size, train_dataset_number, test_dataset_number, batch_size):
    """ Load the BAS dataset from Pennylane. The dataset is composed only composed of 2 classes.
    The MSE version load the labels as tensor of dimension 2 (torch.tensor([1,0]) and torch.tensor([0,1])).
    Args:
        - size: size (4x4, 8x8, 16x16, or 32x32) of the square image (int) 
        - train_dataset_number: number of training samples to keep (int, less than 1000)
        - test_dataset_number: number of testing samples to keep (int, less than 200)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    size_str = str(size)
    [ds] = qml.data.load("other", name="bars-and-stripes")
    x_train = np.array(ds.train[size_str]['inputs']) # vector representations images
    y_train = np.array(ds.train[size_str]['labels']) # labels for the above images  
    y_train_vector = np.zeros((train_dataset_number,2))
    x_test = np.array(ds.test[size_str]['inputs']) # vector representations of images
    y_test = np.array(ds.test[size_str]['labels']) # labels for the above images
    y_test_vector = np.zeros((test_dataset_number,2))
    # Reshape the data:
    x_train, x_test = x_train[:train_dataset_number].reshape(train_dataset_number, size,size), x_test[:test_dataset_number].reshape(test_dataset_number, size,size)
    y_train, y_test = y_train[:train_dataset_number], y_test[:test_dataset_number]
    # Vectorize labels:
    for index, value in enumerate(y_train):
        if value == 1: # Transform label 1 to label torch.tensor([0,1])
            y_train_vector[index][1] = 1
        elif value == -1: # Transform label 0 to label torch.tensor([1,0])
            y_train_vector[index][0] = 1
        else:
            print("Errors with labels")
    for index, value in enumerate(y_test):
        if value == 1: # Transform label 1 to label torch.tensor([0,1])
            y_test_vector[index][1] = 1
        elif value == -1: # Transform label 0 to label torch.tensor([1,0])
            y_test_vector[index][0] = 1
        else:
            print("Errors with labels")
    print("Train data and label tensors of size:{} and {}".format(np.shape(x_train), np.shape(y_train)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(x_test), np.shape(y_test)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train_vector, x_test, y_test_vector, batch_size)

##################################################################################################################
# PCA reduction
##################################################################################################################
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
    (x_train, y_train), (x_test, y_test) = Load_MNIST_tensorflow()
    x_train, x_test = x_train.reshape(x_train.shape[0], 28*28), x_test.reshape(x_test.shape[0], 28*28)
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    y_train, y_test = np.asarray(train_list_label_array), np.asarray(test_list_label_array)
    # Perform PCA reduction:
    x_train, x_test = PCA_reduction(train_list_data_array, test_list_data_array, train_dataset_number, nbr_components)
    x_train, x_test = x_train.reshape(x_train.shape[0], nbr_rows, nbr_cols), x_test.reshape(x_test.shape[0], nbr_rows, nbr_cols)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)

def Digits_sklearn_PCA(class_set, train_dataset_number, test_dataset_number, batch_size, nbr_rows, nbr_cols):
    """ Load the digits dataset from sklearn and apply PCA to reduce the dimensionality.
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
    (x_train, y_train), (x_test, y_test) = Load_Digits_sklearn()
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    y_train, y_test = np.asarray(train_list_label_array), np.asarray(test_list_label_array)
    # Perform PCA reduction:
    x_train, x_test = PCA_reduction(train_list_data_array, test_list_data_array, train_dataset_number, nbr_components)
    x_train, x_test = x_train.reshape(x_train.shape[0], nbr_rows, nbr_cols), x_test.reshape(x_test.shape[0], nbr_rows, nbr_cols)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)


##################################################################################################################
# Average Pooling reduction
##################################################################################################################
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
    (x_train, y_train), (x_test, y_test) = Load_MNIST_tensorflow()
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    y_train, y_test = np.asarray(train_list_label_array), np.asarray(test_list_label_array)
    samples, samples_pooling = np.concatenate((np.array(train_list_data_array), np.array(test_list_data_array)), axis=0), []
    # We use PyTorch AdaptiveAvgPool2d to perform the pooling
    Pooling = AdaptiveAvgPool2d((nbr_rows, nbr_cols))
    for image in samples:
        image = image.reshape(28,28)
        image = torch.tensor(image).unsqueeze(0).float()
        image = Pooling(image)
        samples_pooling.append(image.squeeze(0))
    x_train, x_test = np.asarray(samples_pooling[:train_dataset_number]), np.asarray(samples_pooling[train_dataset_number:])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)

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
    x_train, x_test = x_train.reshape(x_train.shape[0], 8,8), x_test.reshape(x_test.shape[0], 8,8)
    # Only the classes in class_set are kept
    train_list_data_array, train_list_label_array, test_list_data_array, test_list_label_array = select_classes(x_train, y_train, x_test, y_test, train_dataset_number, test_dataset_number, class_set)
    y_train, y_test = np.asarray(train_list_label_array), np.asarray(test_list_label_array)
    samples, samples_pooling = np.concatenate((np.array(train_list_data_array), np.array(test_list_data_array)), axis=0), []
    # We use PyTorch AdaptiveAvgPool2d to perform the pooling
    Pooling = AdaptiveAvgPool2d((nbr_rows, nbr_cols))
    for image in samples:
        image = image.reshape(8,8)
        image = torch.tensor(image).unsqueeze(0).float()
        image = Pooling(image)
        samples_pooling.append(image.squeeze(0))
    x_train, x_test = np.asarray(samples_pooling[:train_dataset_number]), np.asarray(samples_pooling[train_dataset_number:])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)


##################################################################################################################
# BAS Special Preprocess
##################################################################################################################
def BAS_pennylane_special(size, train_dataset_number, test_dataset_number, batch_size):
    """ Load the BAS dataset from Pennylane. The dataset is composed only composed of 2 classes.
    Args:
        - size: size (4x4, 8x8, 16x16, or 32x32) of the square image (int) 
        - train_dataset_number: number of training samples to keep (int, less than 1000)
        - test_dataset_number: number of testing samples to keep (int, less than 200)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    size_str = str(size)
    [ds] = qml.data.load("other", name="bars-and-stripes")
    x_train = np.array(ds.train[size_str]['inputs']) # vector representations images
    y_train = np.array(ds.train[size_str]['labels']) # labels for the above images  
    x_test = np.array(ds.test[size_str]['inputs']) # vector representations of images
    y_test = np.array(ds.test[size_str]['labels']) # labels for the above images
    # Remove all data filled with zeros
    x_train, x_test = x_train.reshape(x_train.shape[0], size, size), x_test.reshape(x_test.shape[0], size, size)
    x_train, x_test = np.where(x_train <-0, -1, 1), np.where(x_test < -0, -1, 1)
    mask_train, mask_test = np.any(x_train != 0, axis=(1, 2)), np.any(x_test != 0, axis=(1, 2))
    x_train, x_test = x_train[mask_train], x_test[mask_test]# Use the mask to filter the 3D array
    #x_train = x_train[~np.all(x_train == 0, axis=(1, 2))]
    #x_test = x_test[~np.all(x_test == 0, axis=(1, 2))]
    # Reshape the data:
    x_train, x_test = x_train[:train_dataset_number], x_test[:test_dataset_number]
    y_train, y_test = y_train[:train_dataset_number], y_test[:test_dataset_number]
    # Transform label -1 to label 0:
    y_train[y_train == -1], y_test[y_test == -1] = 0, 0
    print("Train data and label tensors of size:{} and {}".format(np.shape(x_train), np.shape(y_train)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(x_test), np.shape(y_test)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train, x_test, y_test, batch_size)


##################################################################################################################
# MedMNIST
##################################################################################################################
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def load_medmnist(dataset_name, class_set, train_dataset_number, test_dataset_number, batch_size):
    filtered_train_dataset = load_filtered_medmnist(dataset_name, class_set, train_dataset_number, split='train')
    train_dataloader = create_dataloader(filtered_train_dataset, batch_size)
    filtered_test_dataset = load_filtered_medmnist(dataset_name, class_set, test_dataset_number, split='test')
    test_dataloader = create_dataloader(filtered_test_dataset, batch_size)
    return train_dataloader, test_dataloader

def load_filtered_medmnist(dataset_name, labels_to_include, num_datapoints, split='train'):
    # Get dataset information
    info = INFO[dataset_name] # breastmnist, chestmnist, dermamnist, octmnist, organmnist_axial, organmnist_coronal, organmnist_sagittal, pathmnist, pneumoniamnist, retinamnist
    DataClass = getattr(medmnist, info['python_class'])

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
        #lambda x: F.avg_pool2d(x, kernel_size=2)
    ])

    # Load the MedMNIST dataset
    medmnist_dataset = DataClass(split=split, transform=transform, download=True)

    # Find indices of the datapoints that match the labels_to_include
    indices = [i for i, (img, label) in enumerate(medmnist_dataset) if label in labels_to_include]

    # Shuffle indices to get a random subset
    random_indices = torch.randperm(len(indices)).tolist()

    # Select the specified number of datapoints
    selected_indices = indices[:num_datapoints]

    # Create a subset of the dataset
    filtered_dataset = Subset(medmnist_dataset, selected_indices)

    return filtered_dataset

# Function to create a DataLoader for the filtered dataset
def create_dataloader(filtered_dataset, batch_size=32):
    dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
    return dataloader