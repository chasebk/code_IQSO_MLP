#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Bao Hoang" at 21:43, 01/02/2020                                                           %
#                                                                                                       %
#       Email:      hoangnghiabao96@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Bao_Hoang19                                    %
#       Github:     https://github.com/hoangbao123                                                      %
#-------------------------------------------------------------------------------------------------------%
"""
    Plot stability and convergence speed of 15 runtimes of all algorithm over 30 benmark functions
    Then read input form overall/algo_dict_info.pkl which is generated after run get_experiment_infor.py
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd

def get_loss_fit(fun_index, labels):
    x = np.arange(len(algo_infor['QSO'].loss[1]))
    stable_dict = {}
    loss_dict = {}
    losses = []
    names = []

    for name, al in algo_infor.items():
        if(name in labels):
            if name == 'ABFOLS':
                name = 'ABFO'
            stable_dict.update({name: al.best_fit[fun_index]})
            loss_dict[name] = al.loss[fun_index]

    df_fit = pd.DataFrame(stable_dict)
    df_loss = pd.DataFrame(loss_dict)

    return df_fit, df_loss


def plot_and_save(fun_index, labels_data, labels_legend=None, partly=False):
    file_name = 'F' + str(fun_index + 1)
    if partly:
        file_name += '_partly'
    stable_file = file_name + '_stable.pdf'
    convergence_file = file_name + '_conv.pdf'

    df_fit, df_loss = get_loss_fit(fun_index, labels_data)

    fig1 = plt.figure()
    fig2 = plt.figure()

    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()

    axe1 = sns.lineplot(data=df_loss, ax=ax1, dashes=False)
    axe1.set_xlabel("Iteration")
    axe1.set_ylabel("Fitness")
    if labels_legend is not None:
        axe1.legend(labels_legend)
    else:
        axe1.legend(labels_data)
    axe1.set_title('F' + str(fun_index))

    axe2 = sns.boxplot(data=df_fit, ax=ax2)
    axe2.set_xlabel("Algorithm")
    axe2.set_ylabel("Average Fitness")
    axe2.set_title(file_name)

    fig1.savefig('./history/convergence/' + convergence_file)
    fig2.savefig("./history/stability/" + stable_file, figsize=(1, 2), dpi=10)
    # plt.show()
    fig1.clf()
    fig2.clf()
    # fig1.close()
    # fig2.close()


if __name__ == '__main__':
    labels_to_get_data = ['GA', 'ABC', 'PSO', 'CRO', 'WOA', 'QSO', 'IQSO']
    labels_to_plot_legend = ['GA', 'ABC', 'PSO', 'CRO', 'WOA', 'QSO', 'IQSO']
    labels_to_get_partly_data = ['WOA', 'QSO', 'IQSO']
    with open('./history/overall/algo_dict_info.pkl', 'rb') as f:
            algo_infor = pkl.load(f)
    partly = True
    if partly is False:
        for i in range(30):
            plot_and_save(i, labels_to_get_data, labels_to_plot_legend)
    else:
        for i in range(30):
            plot_and_save(i, labels_to_get_partly_data, partly=True)
    # print(algo_infor['PSO'].name)
