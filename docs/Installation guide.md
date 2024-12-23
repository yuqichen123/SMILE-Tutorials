# Installation guide

The SMILE package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/) and [PyG](https://github.com/pyg-team/pytorch_geometric) (PyTorch Geometric) framework.

## Step 1: Download SMILE
First clone the repository.

```python
git clone https://github.com/lhzhanglabtools/SMILE.git
cd SMILE-main
```
## Step 2: Prepare environment
 We recommend using the [Anaconda Python Distribution](https://anaconda.org/) and creating an isolated environment, so that the SMILE and dependencies don't conflict or interfere with other packages or applications. To create the environment, run the following script in command line:
```python
#create an environment called env_SMILE
conda create -n env_SMILE python=3.11

#activate your environment
conda activate env_SMILE
```

## Step 3: Install relevant packages
Install all the required packages. The torch-geometric library is required, please see the installation steps in <https://github.com/pyg-team/pytorch_geometric#installation>

```python
conda install pyg
conda install conda-forge::pytorch_scatter
conda install conda-forge::pytorch_cluster
conda install conda-forge::pytorch_sparse
```

The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See <https://pypi.org/project/rpy2/> and https://cran.r-project.org/web/packages/mclust/index.html for detail.

```python
pip install -r requirements.txt
```
## Step 4: Install SMILE
Install SMILE. We provide two optional strategies to install SMILE.
```python
pip install stSMILE
```
Or 
```python
python setup.py build
python setup.py install
```

### Other 

### Install pytorch
Please choose the appropriate PyTorch version based on your Python version, CUDA version, and other relevant parameters. Refer to the details provided by [Pytorch](https://pytorch.org/). For example:

```python
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Install R
This project includes dependencies that require the R environment. Please refer to the official [R](https://cran.r-project.org) website for installation instructions. For example, the installation steps on Ubuntu are as follows:

```python
# update indices
sudo apt update -qq
# install two helper packages we need
sudo apt install --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

```


