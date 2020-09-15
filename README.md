# Introduction
The python code in **python** folder implements the simulations and plots the figures described in the paper "Deep Completion Autoencoders for Radio Map Estimation" by Yves Teganya and Daniel Romero.

# Required packages

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
tensorflow
scipy
cvxpy
cvxopt
matplotlib
pandas
joblib
scikit-dsp-comm
sklearn
opencv-python
```
# Guidelines
Add the simulation environment for Python with the following command:

```git submodule add https://github.com/fachu000/GSim-Python.git ./gsim```
 

The first time one wants to run a simulation after downloading the required python packages and the simulation environment, one enters the folder **python**. After that, one will be able to execute any simulation in the aforementioned paper by running `run_experiment.py` that is located  in the folder **python/** followed by the experiment number as will be explained with an example below.

The experiments reproducing different figures in the paper are organized in methods located in the file `Experiments/autoencoder_experiments.py`. The comments before each method indicate which figure(s) on the paper it generates.

For experiments that use the Wireless Insite software, please download the data set at [here](https://pip.pypa.io/en/stable/) and the place the **remcom_maps** folder in the **Generators**. 

One is now all set. For example, to run experiment 1003, one types `run_experiment.py 1003`. To just display the results of the last execution of experiment 1003 (stored in **output/autoencoder_experiments**), one types `run_experiment.py -p 1003`. Note that this way of only ploting executed experiments works for displayinh 1D curves, it does not work for diplaying 2D images. The simulation results for experiments that produce 2D images (possibly together with 1D curves) are saved in **output/autoencoder_experiments/savedResults**. 

# Citation
If our code is helpful in your resarch or work, please cite our paper.
```bash
@article{teganya2020deepcompletion,
  title={Deep Completion Autoencoders for Radio Map Estimation},
  author={Teganya, Yves and Romero, Daniel},
  journal={arXiv preprint arXiv:2005.05964},
  year={2020}
}
```
