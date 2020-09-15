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

The first time one wants to run a simulation after downloading the required python packages and the simulation environment, one enters the folder **python**. After that, one will be able to execute any simulation in the aforementioned paper by running `run_experiment.py` that is located  in the folder **python/**.

The experiments reproducing different figures in the paper are organized in methods located in the file Experiments/LocFCartogrExperiments.m. The comments before each method indicate which figure(s) on the paper it generates.

One is now all set. For example, to run experiment 401 with 100 iterations, one types gsim (0, 401, 100). To just display the results of the last execution of experiment 401 (stored in Experiments/LocFCartogrExperiments_data), one types gsim(1, 401).
