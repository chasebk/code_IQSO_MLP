from models.swarm.ABC import BaseABC
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C20
}
abc_paras = {
    "epoch": 500,
    "pop_size": 100,
    "couple_bees": [16, 4],               # number of bees which provided for good location and other location
    "patch_variables": [5.0, 0.985],        # patch_variables = patch_variables * patch_factor (0.985)
    "sites": [3, 1],                        # 3 bees (employed bees, onlookers and scouts), 1 good partition
}

# Run model
md = BaseABC(root_algo_paras=root_paras, abc_paras=abc_paras)
a, b, c = md._train__()
print(c)
