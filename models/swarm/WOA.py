import numpy as np
from models.root_algo import RootAlgo

class BaseWOA(RootAlgo):
    """
        Standard version of Whale Optimization Algorithm (belongs to Swarm-based Algorithms)
        - In this algorithms: Prey means the best solution
    """
    def __init__(self, root_algo_paras=None, woa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        for i in range(self.epoch):
            a = 2 - 2 * i / (self.epoch - 1)            # linearly decreased from 2 to 0

            for j in range(self.pop_size):

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = np.random.rand()
                b = 1

                if (p < 0.5) :
                    if np.abs(A) < 1:
                        D = np.abs(C * g_best[self.ID_POS] - pop[j][self.ID_POS] )
                        new_position = g_best[0] - A * D
                    else :
                        x_rand = pop[np.random.randint(self.pop_size)]              # Select 1 random solution in pop
                        D = np.abs(C * x_rand[self.ID_POS] - pop[j][self.ID_POS])
                        new_position = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = np.abs(g_best[0] - pop[j][0])
                    new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best[self.ID_POS]

                self._amend_solution__(new_position)
                fit = self._fitness_model__(new_position)
                pop[j] = [new_position, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROBLEM, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(i + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train, g_best[self.ID_FIT]
