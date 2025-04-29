# Photonic_Simulation_QCNN
Simulation library for the photonic implementation of a Subspace Preserving Quantum Neural Network. This toolkit allows to perform efficient Pytorch-based simulation of linear optical and adaptivity quantum circuit by only considering appropriate subspaces during the computation.


Warning! The quantum optics library `qoptcraft` does not support `python>=3.12`, so use version `3.11`.

Example setup commands that should work:

```shell
conda create -n qml python=3.11
conda activate qml
pip install jupyter qoptcraft seaborn

# Then install pytorch however you like
# (see the helper at https://pytorch.org/), e.g. for linux-cpuonly: 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Finally, open the notebook with:
jupyter-lab
```
