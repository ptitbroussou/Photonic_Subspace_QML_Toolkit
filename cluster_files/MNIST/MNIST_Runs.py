### Loading the required libraries
import os, sys
import torch

print("WARNING. Is torch.cuda.is_available():",torch.cuda.is_available())

# Fet the absolute path to the root of the Repo (Photonic_Simulation_QCNN):
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Add repo root to the system path:
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Data_Loading.Classical_Data import *

# Hyperparameters:
batch_size = 1
train_dataset_number = 500 #546
test_dataset_number = 200 #156
class_set = [i for i in range(2)]

# Tensorflow MNIST dataset:
print('Tensorflow MNIST dataset:')
train_data_loader, test_dataloader = Digits_sklearn_raw(class_set, train_dataset_number, test_dataset_number, batch_size)


# Define the Quantum CNN model:
from qoptcraft.basis import get_photon_basis, hilbert_dim
from Layers.Linear_Optics import *
from Layers.measurement import *
from Layers.toolbox_basis_change import Basis_Change_Image_to_Fock_density, Basis_Change_Image_to_larger_Fock_density
from Layers.HW_preserving_QCNN.Conv import Conv_RBS_density_I2
from Layers.HW_preserving_QCNN.Pooling import Pooling_2D_density_HW
from Layers.toolbox import PQNN

### Hyperparameters:
m = 8+8 #number of modes in total
add1, add2 = 0, 4
nbr_class = 2
list_gates = PQNN(m//4 + add1 + add2)
modes_detected = [i for i in range(len(class_set))]
device = torch.device("cpu")

photon_basis = get_photon_basis(m, 2)
dimension = hilbert_dim(m, 2)
print("Hilbert space dimension: ", dimension)


class Photonic_QCNN(nn.Module):
    """ A class defining the fully connected neural network """
    def __init__(self, m, list_gates, modes_detected, device):
        super(Photonic_QCNN, self).__init__()
        self.device = device
        self.Conv1 = Conv_RBS_density_I2(8, 2, device)
        self.Pool1 = Pooling_2D_density_HW(8, 4, device)
        self.toFock = Basis_Change_Image_to_larger_Fock_density(4,4, add1, add2, device)
        #self.toFock = Basis_Change_Image_to_Fock_density(7, 7, device)
        self.dense = VQC_Fock_BS_density(2, 8+add1+add2, list_gates,device)
        self.measure = Measure_Photon_detection(2, 8+add1+add2, modes_detected, device)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Pool1(x)
        return self.measure(self.dense(self.toFock(x)))
       


### Training the network:
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from Training.training import train_globally


### We run multiple time to have average results and variance:
nbr_test = 5
training_loss, training_acc, testing_loss, testing_acc = [], [], [], []
for test in range(nbr_test):
    print("TEST NUMBER {}/{}".format(test+1, nbr_test))
    # Define the network:
    network_dense = Photonic_QCNN(m, list_gates, modes_detected, device)
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(network_dense.parameters())
    optimizer = torch.optim.Adam(network_dense.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    output_scale, train_epochs, test_interval = 5, 15, 1

    network_state, training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list = train_globally(batch_size, m//2, network_dense, train_data_loader, test_dataloader, optimizer, scheduler,
                                  criterion, output_scale, train_epochs, test_interval, device)
    torch.save(network_state, "model_state_MNIST_{}".format(test))  # save network parameters
    training_data = torch.tensor([training_loss_list, training_accuracy_list, testing_loss_list, testing_accuracy_list])
    torch.save(training_data, 'MNIST_training_data_{}.pt'.format(test))
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
