import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Networks import Trainer
import Preprocessing


def show_graph(data):
    indexes = pd.MultiIndex.from_product([['Min', 'Max', 'Mean'], ['Inflow', 'Outflow']], names=['Measure', 'Type'])
    test = np.hstack([np.min(data, axis=(2, 3)), np.max(data, axis=(2, 3)), np.mean(data, axis=(2, 3))])
    data_summary = pd.DataFrame(test, columns=indexes)
    data_summary.plot()
    plt.show()
    return data_summary

def group_info(DB_names, nets_names, nbs_Modules, filters_size, activations, save_path):
    information = []
    for DB_name in DB_names:
        save_path = os.path.join(save_path, DB_name)
        for net_name in nets_names:
            for activation in activations:
                for filter_size in filters_size:
                    for nb_Modules in nbs_Modules:
                            info = Trainer.reader(DB_name, net_name, nb_Modules, filter_size, activation, save_path)
                            information.append(info)
    indexes = pd.MultiIndex.from_product([DB_names, nets_names, nbs_Modules, filters_size, activations, ['Acc', 'Loss']], names=['DB_names', 'nets_names', 'nbs_Modules', 'filters_size', 'activations', 'type'])
    print(indexes)
    information = np.hstack(information)
    data_summary = pd.DataFrame(information, columns=indexes)
    return data_summary


def plot_info(data_summary, value=None, level=None):
    if value != None and level != None:
        data_summary.xs(value, level=level, axis=1).plot()
    else:
        data_summary.plot()
    plt.show()


if __name__ == '__main__':
    DB_names = ['BikeNYC']
    activations = ['tanh']
    filters_size = [16]
    nbs_Modules = [1]
    nets_names = ['ConvLSTM_Inception_ResNet_type_1']
    len_closeness = 12
    len_period = 1
    len_trend = 12
    save_path = os.path.join(os.getcwd(), 'tmp')
    #data_summary = group_info(DB_names, nets_names, nbs_Modules, filters_size, activations, save_path)
   # plot_info(data_summary, level=('nbs_Modules', 'type'), value=(1, 'Acc'))
    DB_names = ['BikeNYC']
    for DB_name in DB_names:
        filePath = os.path.join(os.getcwd(), 'DB/databases_Cleaned/paper_preprocess/{}'.format(DB_name), 'len_closeness_{}_len_period_{}_len_trend_{}'.format(len_closeness, len_period, len_trend))
        X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(DB_name=DB_name, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        show_graph(Y_train)
        show_graph(Y_test)