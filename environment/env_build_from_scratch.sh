#!/bin/bash

envname=$1

conda deactivate

conda create -y -n $envname python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda=11.7 numba -c pytorch -c nvidia -c conda-forge

conda activate $envname

# install ffcv
pip install ffcv

# packages for testing environment in jupyterlab
mamba install ipykernel ipython jupyterlab ipywidgets

# add kernel to jupyterlab
ipython kernel install --user --name=$envname

# some packages for github/jupyter integration
mamba install GitPython nbstripout nbconvert
nbstripout --install

# wandb integration
mamba install wandb
wandb login # will need to manually paste in key

# for dealing with COCO dataset
mamba install pycocotools

# for running torchlens
pip install graphviz
pip install git+https://github.com/johnmarktaylor91/torchlens

# other packages for analyses
mamba install scipy torchmetrics seaborn nibabel h5py

# install this project package
pip install --user -e ../

# pycortex install
pip install pycortex

# set directory of the pycortex database
python3 pycortex_database_setup.py

# install circuit pruning tools
mamba install umap-learn=0.5.3
mamba install dash=2.7.1
mamba install jupyter-dash=0.4.2
pip install lucent==0.1.0
pip install kornia==0.4.1
pip install kaleido==0.2.1
mamba install pyarrow=10.0.1

pip install --user -e /home/jovyan/work/DropboxSandbox/circuit_pruner_iccv2023/circuit_pruner_code

# test imports
python3 env_test_imports.py
