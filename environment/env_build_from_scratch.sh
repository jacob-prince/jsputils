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

# test imports
python3 env_test_imports.py

# set directory of the pycortex database
python3 pycortex_database_setup.py
