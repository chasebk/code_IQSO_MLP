from random import random, randint
from numpy import maximum, minimum
from copy import deepcopy
from models.root_algo import RootAlgo

class BaseABC(RootAlgo):
    """
    Taken from book: Clever Algorithms
        - Improved function _create_neigh_bee__
        - Better results, faster convergence
    """

    def __init__(self, root_algo_paras=None, abc_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = abc_paras["epoch"]
        self.pop_size = abc_paras["pop_size"]
        self.e_bees = abc_paras["couple_bees"][0]
        self.o_bees = abc_paras["couple_bees"][1]

        self.patch_size = abc_paras["patch_variables"][0]
        self.patch_factor = abc_paras["patch_variables"][1]
        self.num_sites = abc_paras["sites"][0]
        self.elite_sites = abc_paras["sites"][1]

    def _create_neigh_bee__(self, individual=None, patch_size=None):
        t1 = randint(0, len(individual) - 1)
        new_bee = deepcopy(individual)
        new_bee[t1] = (individual[t1] + random() * patch_size) if random() < 0.5 else (individual[t1] - random() * patch_size)
        new_bee[t1] = maximum(self.domain_range[0], minimum(self.domain_range[1], new_bee[t1]))
        return [new_bee, self._fitness_model__(new_bee)]


    def _search_neigh__(self, parent=None, neigh_size=None):  # parent:  [ vector_individual, fitness ]
        """
        Search 1 best solution in neigh_size solution
        """
        neigh = [self._create_neigh_bee__(parent[self.ID_POS], self.patch_size) for _ in range(0, neigh_size)]
        return self._get_global_best__(neigh, self.ID_FIT, self.ID_MIN_PROBLEM)

    def _create_scout_bees__(self, num_scouts=None):
        return [self._create_solution__() for _ in range(0, num_scouts)]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        pop = sorted(pop, key=lambda item: item[self.ID_FIT])
        g_best = deepcopy(pop[self.ID_MIN_PROBLEM])

        for epoch in range(0, self.epoch):
            next_gen = []
            for i in range(0, self.num_sites):
                if i < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                next_gen.append(self._search_neigh__(pop[i], neigh_size))

            scouts = self._create_scout_bees__(self.pop_size - self.num_sites)
            pop = next_gen + scouts

            ## sort pop and update global best
            g_best, pop = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROBLEM, g_best)
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, patch_size = {}, Fit = {}".format(epoch + 1, self.patch_size, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train

