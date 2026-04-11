from dataclasses import dataclass

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE DO NOT EDIT/MODIFY THIS FILE
# If you want to use other argument values than the defaults, specify them
# in the command line. For instance: 
# To launch the program with 2 runs, just run:
#
#           accelerate launch runproject.py --n_runs=2
#

@dataclass
class Config:
    """
    Dataclass for the script arguments
    """
    # General options
    ollama_url: str = 'http://localhost:11434/v1'
    ollama_url: str = "http://chaos-04.int.europe.naverlabs.com:11434/v1"
    ollama_model: str = "gemma3:4b"
    #
    eval_batch_size: int = 10
    n_runs: int = 5
    # n_train is the number of samples on which to train. n_train=-1 means train all train data
    n_train: int = -1
    # n_eval is the number of samples on which to run the eval. n_eval=-1 means eval on all data samples
    n_eval: int = -1



