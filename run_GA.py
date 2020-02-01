from models.evolution.GA import BaseGA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C11
}
ga_paras = {
    "epoch": 1000,
    "pop_size": 100,
    "pc": 0.95,
    "pm": 0.025
}

## Run model
md = BaseGA(root_algo_paras=root_paras, ga_paras=ga_paras)
a, b, c = md._train__()
print(b)
print(c)
