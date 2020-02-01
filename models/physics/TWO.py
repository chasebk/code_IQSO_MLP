#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Bao Hoang" at 12:43, 27/11/2020                                                           %
#                                                                                                       %
#       Email:      hoangnghiabao96@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Bao_Hoang19                                    %
#       Github:     https://github.com/hoangbao123                                                      %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from models.root_algo import RootAlgo

class BaseTWO(RootAlgo):
    ID_POS = 0
    ID_FIT = 1
    ID_WEIGHT = 2

    def __init__(self, root_algo_paras=None, two_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = two_paras["epoch"]
        self.pop_size = two_paras["pop_size"]

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        weight = 0.0
        return [solution, fitness, weight]

    def _update_weight__(self, teams):
        best_fitness = max(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        worst_fitness = min(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        for i in range(self.pop_size):
            teams[i][self.ID_WEIGHT] = (teams[i][self.ID_FIT] - worst_fitness)/(best_fitness - worst_fitness) + 1
        return teams

    def _update_fit__(self, teams):
        for i in range(self.pop_size):
            teams[i][self.ID_FIT] = self._fitness_model__(teams[i][self.ID_POS], minmax=self.ID_MAX_PROBLEM)
        return teams

    def _amend_and_return_pop__(self, pop_old, pop_new, g_best, epoch):
        for i in range(self.pop_size):
            for j in range(self.problem_size):
                if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                    if np.random.random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = g_best[self.ID_POS][j] + np.random.randn()/(epoch+1)*(g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                            pop_new[i][self.ID_POS][j] = pop_old[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[0]
                        if pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[1]
        return pop_new

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range( self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                       force = max(pop_old[i][self.ID_WEIGHT]*muy_s, pop_old[j][self.ID_WEIGHT]*muy_s)
                       resultant_force = force - pop_old[i][self.ID_WEIGHT]*muy_k
                       g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                       acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*muy_k)
                       delta_x = 1/2*acceleration + np.power(alpha,epoch+1)\
                                 *beta*(self.domain_range[1] -  self.domain_range[0])*\
                                np.random.randn(self.problem_size)
                       pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(np.power(g_best[self.ID_FIT], -1))
            if self.print_train:
                print("Epoch: {}, best result so far: {}".format(epoch + 1, np.power(g_best[self.ID_FIT], -1)))
        return g_best[self.ID_POS], self.loss_train, g_best[self.ID_FIT], g_best[self.ID_FIT]
