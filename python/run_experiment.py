# Python 3.6
#

import sys
import os
# from IPython.core.debugger import set_trace


def initialize():
    if not os.path.isfile("./run_experiment.py"):
        # If this behavior were changed, the output file storage functionality should be
        # modified accordingly.
        print(
            "Error: `run_experiment` must be invoked from the folder where it is defined"
        )
        quit()

    sys.path.insert(1, './gsim/')


initialize()
########################################################################
# Select experiment file:
from Experiments.autoencoder_experiments import ExperimentSet

########################################################################

if (len(sys.argv) < 2):
    print(
        'Usage: python3 ', sys.argv[0],
        '[option] <experiment_index> [cmdl_arg1 [cmdl_arg2 ... [cmdl_argN]]]')
    print("""       <experiment_index>: identifier for the experiment 
             
            cmdl_argn: n-th argument to the experiment function (optional) 

            OPTIONS: 
            
            -p : plot only the stored results, do not run the simulations.

    """)
    quit()

l_args = sys.argv
if l_args[1] == "-p":
    # Plot only
    ExperimentSet.plot_only(l_args[2])

else:

    if (len(l_args) < 3):
        cmdl_args = ""
    else:
        cmdl_args = l_args[2:]

    ExperimentSet.run_experiment(l_args[1], cmdl_args)

# if 0:
#     logging.basicConfig(level=logging.INFO, filename ='LogFile_%s.txt' /
#                         % experiment_number)
#     sys.stdout = open('LogFile_%s.txt' % experiment_number, 'a')
#     sys.stderr = open('LogFile_%s.txt' % experiment_number, 'a')
# experiment_name = 'run_experiment_%s'%experiment_number
