from models.physics.TWO import OppoTWO
from utils.FunctionUtil import *

root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
two_paras = {
    "epoch": 1000,
    "pop_size": 100,
}

## Run model
md = OppoTWO(root_algo_paras=root_paras, two_paras=two_paras)
md._train__()
