### Loading the required libraries
import os, sys
import torch
import pickle


print("WARNING. Is torch.cuda.is_available():",torch.cuda.is_available())

# Fet the absolute path to the root of the Repo (Photonic_Simulation_QCNN):
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Add repo root to the system path:
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


from Data_Loading.Classical_Data import *

with open("FULL_DATASET_600_samples.bin", "rb") as f:
    data_02 = pickle.load(f)

from Data_Loading.toolbox_data import create_pytorch_dataset

batch_size = 1
train_dataset_number = 400
test_dataset_number = 200

# Split the data into training and test sets
def BAS_Custom_MSE(data, train_dataset_number, test_dataset_number, batch_size):
    """ Load the BAS dataset from Pennylane. The dataset is composed only composed of 2 classes.
    Args:
        - data: the dataset to be used
        - train_dataset_number: number of training samples to keep (int, less than 1000)
        - test_dataset_number: number of testing samples to keep (int, less than 200)
        - batch_size: size of the batch (int)
    Output:
        - train_dataloader: training dataloader
        - test_dataloader: testing dataloader
    """
    x_train = np.array([data[i][1] for i in range(train_dataset_number)]) # vector representations images
    y_train = np.array([data[i][0] for i in range(train_dataset_number)]) # labels for the above images  
    x_test = np.array([data[i][1] for i in range(train_dataset_number, train_dataset_number+test_dataset_number)]) # vector representations of images
    y_test = np.array([data[i][0] for i in range(train_dataset_number, train_dataset_number+test_dataset_number)]) # labels for the above images
    y_train_vector, y_test_vector = np.zeros((train_dataset_number,2)), np.zeros((test_dataset_number,2)) # labels for the above images
    # Reshape the data:
    x_train, x_test = x_train[:train_dataset_number].reshape(train_dataset_number, 4,4), x_test[:test_dataset_number].reshape(test_dataset_number, 4,4)
    y_train, y_test = y_train[:train_dataset_number], y_test[:test_dataset_number]
    # Vectorize labels:
    for index, value in enumerate(y_train):
        if value == 1: # Transform label 1 to label torch.tensor([0,1])
            y_train_vector[index][1] = 1
        elif value == 0: # Transform label 0 to label torch.tensor([1,0])
            y_train_vector[index][0] = 1
        else:
            print("Errors with labels")
    for index, value in enumerate(y_test):
        if value == 1: # Transform label 1 to label torch.tensor([0,1])
            y_test_vector[index][1] = 1
        elif value == 0: # Transform label 0 to label torch.tensor([1,0])
            y_test_vector[index][0] = 1
        else:
            print("Errors with labels")
    print("Train data and label tensors of size:{} and {}".format(np.shape(x_train), np.shape(y_train_vector)))
    print("Test data and label tensors of size:{} and {}".format(np.shape(x_test), np.shape(y_test_vector)))
    # Convert numpy arrays to PyTorch tensors and form dataloaders
    return create_pytorch_dataset(x_train, y_train_vector, x_test, y_test_vector, batch_size)

# Load the data
train_dataloader_02, test_dataloader_02 = BAS_Custom_MSE(data_02, train_dataset_number, test_dataset_number, batch_size)






# Define the Quantum CNN model:
from qoptcraft.basis import get_photon_basis, hilbert_dim
from Layers.Linear_Optics import *
from Layers.measurement import *
from Layers.toolbox_basis_change import Basis_Change_Image_to_Fock_density, Basis_Change_Image_to_larger_Fock_density
from Layers.HW_preserving_QCNN.Conv import Conv_RBS_density_I2
from Layers.HW_preserving_QCNN.Pooling import Pooling_2D_density_HW
from Layers.toolbox import PQNN

### Hyperparameters:
m = 4 + 4 #number of modes in total
add1, add2 = 1, 1
nbr_class = 2
#list_gates = [(2,3),(1,2),(3,4),(0,1),(2,3),(4,5),(1,2),(3,4),(0,1),(2,3),(4,5),(1,2),(3,4)]
list_gates = [(2,3),(1,2),(3,4),(0,1),(2,3),(4,5),(1,2),(3,4)]
modes_detected = [2,3]


device = torch.device("cpu")


class Photonic_QCNN(nn.Module):
    """ A class defining the fully connected neural network """
    def __init__(self, m, list_gates, modes_detected, device):
        super(Photonic_QCNN, self).__init__()
        self.device = device
        self.Conv = Conv_RBS_density_I2(m//2, 2, device)
        self.Pool = Pooling_2D_density_HW(m//2, m//4, device)
        self.toFock = Basis_Change_Image_to_larger_Fock_density(m//4, m//4, add1, add2, device)
        self.dense = VQC_Fock_BS_density(2, m//2 + add1 + add2, list_gates,device)
        self.measure = Measure_Photon_detection(2, m//2 + add1 + add2, modes_detected, device)
        #self.toFock =  Basis_Change_Image_to_Fock_density(m//4,m//4,device)
        #self.dense = VQC_Fock_BS_density(2, m//2, list_gates,device)
        #self.measure = Measure_Photon_detection(2, m//2, 0, device)
    def forward(self, x):
        x = self.Conv(x)
        x = self.Pool(x)
        return self.measure(self.dense(self.toFock(x)))
       

for (data, target) in train_dataloader_02:
    print("Data shape: ", data.shape)
    print("Target shape: ", target.shape)
    break



### Training the network:
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from Training.training import train_globally_MSE, train_globally

# Define the network:
criterion = torch.nn.MSELoss()
### We run multiple time to have average results and variance:
nbr_test = 5
training_loss, training_acc, testing_loss, testing_acc = [], [], [], []
for test in range(nbr_test):
    print("Number of test {}/{}".format(test, nbr_test))
    # Define the network:
    network_dense = Photonic_QCNN(m, list_gates, modes_detected, device)
    criterion = torch.nn.MSELoss()

    #optimizer = torch.optim.SGD(network_dense.parameters())
    optimizer = torch.optim.Adam(network_dense.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    output_scale, train_epochs, test_interval = 1, 30, 1

    network_state, training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list = train_globally_MSE(batch_size, 4, network_dense, train_dataloader_02, test_dataloader_02, optimizer, scheduler,
                                  criterion, output_scale, train_epochs, test_interval, device)
    torch.save(network_state, "model_state_Custom_BAS_{}".format(test))  # save network parameters
    training_data = torch.tensor([training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list])
    torch.save(training_data, 'Custom_BAS_training_data_{}.pt'.format(test))
    # saving the data to perform expectation and average values:
    training_loss.append(training_loss_list)
    training_acc.append(training_accuracy_list)
    testing_loss.append(testing_loss_list)
    testing_acc.append(testing_accuracy_list)


# Average and Standard deviation calculus:
average_train_loss = np.array([sum([training_loss[i][j] for i in range(nbr_test)])/nbr_test for j in range(train_epochs)])
average_train_acc = np.array([sum([training_acc[i][j] for i in range(nbr_test)])/nbr_test for j in range(train_epochs)])
average_test_loss = np.array([sum([testing_loss[i][j] for i in range(nbr_test)])/nbr_test for j in range(train_epochs)])
average_test_acc = np.array([sum([testing_acc[i][j] for i in range(nbr_test)])/nbr_test for j in range(train_epochs)])

std_train_loss = np.array([np.sqrt(sum([training_loss[i][j]**2 - average_train_loss[j] for i in range(nbr_test)])/nbr_test) for j in range(train_epochs)])
std_train_acc = np.array([np.sqrt(sum([training_acc[i][j]**2 - average_train_acc[j] for i in range(nbr_test)])/nbr_test) for j in range(train_epochs)])
std_test_loss = np.array([np.sqrt(sum([testing_loss[i][j]**2 - average_test_loss[j] for i in range(nbr_test)])/nbr_test) for j in range(train_epochs)])
std_test_acc = np.array([np.sqrt(sum([testing_acc[i][j]**2 - average_test_acc[j] for i in range(nbr_test)])/nbr_test) for j in range(train_epochs)])
