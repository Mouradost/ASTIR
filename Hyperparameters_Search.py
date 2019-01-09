import tensorflow as tf
import Preprocessing
import os
from Networks import ConvLSTM_Inception_ResNet, Trainer
import skopt
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import pandas as pd
tf.set_random_seed(1337)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nbr_gpu = 1

DB_name = 'BikeNYC'
len_closeness = skopt.space.Integer(low=1, high=5, name='len_closeness')
len_period = skopt.space.Integer(low=1, high=5, name='len_period')
len_trend = skopt.space.Integer(low=1, high=5, name='len_trend')
lr = skopt.space.Real(low=5e-3, high=5e-2, prior='log-uniform', name='lr')
net_type = skopt.space.Categorical(categories=[0, 1, 2], name='net_type')
nb_Modules = skopt.space.Integer(low=1, high=2, name='nb_Modules')
filter_size = skopt.space.Categorical(categories=[8, 16, 32], name='filter_size')
activation = skopt.space.Categorical(categories=['relu', 'tanh'], name='activation')
dropout = skopt.space.Real(low=0, high=0.9, name='dropout')
l1 = skopt.space.Real(low=0, high=0.05, name='l1')
l2 = skopt.space.Real(low=0, high=0.05, name='l2')
dimensions = [len_closeness, len_period, len_trend, lr, net_type, nb_Modules, filter_size, activation, dropout, l1, l2]

@skopt.utils.use_named_args(dimensions=dimensions)
def fitness(len_closeness, len_period, len_trend, lr, net_type, nb_Modules, filter_size, activation, dropout, l1, l2):

    net_name = 'ConvLSTM_Inception_ResNet'
    epoch = 100
    batch_size = 32
    decay = 0.0
    load_weights = False
    save_weight = False
    print('*'*100)
    print('*' * 50)
    print('Tensorflow version : {}'.format(tf.VERSION))
    print('_' * 50)
    print(
        'Database name : {}\nModel type : {}\nActivation : {}\nNumber of layers : {}\nNumber of filters : {}'.format(
            DB_name, net_name, activation, nb_Modules, filter_size))
    print('_' * 50)
    print(
        'Closeness lent : {}\nPeriod lent : {}\nTemporal lent : {}\n'.format(
            len_closeness, len_period, len_trend))
    print('_' * 50)
    print(
        'Epoch : {}\nBatch size : {}\nLearning rate : {}\nDecray : {}\nDropout : {}'.format(
            epoch, batch_size, lr, decay, dropout))
    print('*' * 50)
    try:
        if DB_name == 'TaxiBJ':
            c_conf = (len_closeness, 2, 32, 32)
            p_conf = (len_period, 2, 32, 32)
            t_conf = (len_trend, 2, 32, 32)
            output_shape = (2, 32, 32)
            external_shape = (28,)
            X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(DB_name, len_closeness, len_period, len_trend)
            X_train, X_test = Preprocessing.prepare_data_as_a_sequence(X_train, X_test, len_closeness=len_closeness,
                                                                       len_period=len_period, len_trend=len_trend,
                                                                       channel=output_shape[0])

        else:
            c_conf = (len_closeness, 2, 16, 8)
            p_conf = (len_period, 2, 16, 8)
            t_conf = (len_trend, 2, 16, 8)
            output_shape = (2, 16, 8)
            external_shape = ()
            X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(DB_name, len_closeness, len_period, len_trend)
            X_train, X_test = Preprocessing.prepare_data_as_a_sequence(X_train, X_test, len_closeness=len_closeness,
                                                                       len_period=len_period, len_trend=len_trend,
                                                                       channel=output_shape[0])
        save_path = os.path.join(os.getcwd(), 'tmp', '{}'.format(DB_name))
        save_path_tmp = os.path.join(save_path, 'Lr={0:.5}'.format(lr))
        save_path_tmp = os.path.join(save_path_tmp,
                                     'len_closeness={}_len_period={}_len_trend={}'.format(len_closeness, len_period,
                                                                                          len_trend))
        os.makedirs(save_path_tmp, exist_ok=True)

        model = ConvLSTM_Inception_ResNet.convLSTM_Inception_ResNet_network(c_conf=c_conf, p_conf=p_conf,
                                                                            t_conf=t_conf,
                                                                            output_shape=output_shape,
                                                                            external_shape=external_shape,
                                                                            nb_modules=nb_Modules,
                                                                            filters=filter_size, kernel_size=(3, 3),
                                                                            strides=(1, 1), padding='same',
                                                                            data_format='channels_first',
                                                                            activation=activation, dropout=dropout,
                                                                            l1=l1, l2=l2,
                                                                            types=net_type)
        network_name = '{}_Type_{},l1={},l2={},dropout={}'.format(net_name, net_type, l1, l2, dropout)
        print('*' * 50)
        print('Tensorflow version : {}'.format(tf.VERSION))
        print('Keras version : {}'.format(tf.keras.__version__))
        print('_' * 50)
        print(
            'Database name : {}\nModel type : {}\nActivation : {}\nNumber of layers : {}\nNumber of filters : {}'.format(
                DB_name, network_name, activation, nb_Modules, filter_size))
        print('_' * 50)
        print(
            'Closeness shape : {}\nPeriod shape : {}\nTemporal shape : {}\nOutput shape : {}\nExternals shape : {}'.format(
                c_conf, p_conf, t_conf, output_shape, external_shape))
        print('_' * 50)
        print(
            'Epoch : {}\nBatch size : {}\nLearning rate : {}\nDecray : {}\nDropout : {}'.format(
                epoch, batch_size, lr, decay, dropout))
        print('*' * 50)

        if nbr_gpu > 1:
            try:
                model = tf.keras.utils.multi_gpu_model(model, gpus=nbr_gpu)
                print("Training using multiple GPUs..")
            except Exception as e:
                print("Training using single GPU..")
                print("Error : ", e)

        info, score = Trainer.train(model=model, X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test, mmn=mmn,
                                    DB_name=DB_name, epochs=epoch, batch_size=batch_size, network_name=network_name,
                                    nb_modules=nb_Modules, filters=filter_size, activation=activation, learning_rate=lr,
                                    decay=decay, load_weights=load_weights, save_weight=save_weight,
                                    save_path=save_path_tmp)
        metric = score['RMSE_Real']
        with open(os.path.join(save_path, 'report.txt'), 'a+') as f:
            f.writelines(
                'Network_Name:{},Type:{},Activation:{},Nb_Modules:{},Filter_Size:{},l1:{},l2:{},Learning_Rate:{:.5f},Dropout:{:.5f},Closeness:{},Period:{},Trend:{}/Train_MSE:{:.6f},Train_Accuracy:{:.2%},RMSE(norm):{:.6f},RMSE(real):{:.4f},MAPE:{:.2f}%\n'.format(
                net_name, net_type, activation, nb_Modules, filter_size, l1, l2, lr, dropout, len_closeness, len_period,
                len_trend, score['MSE_Train'], score['Accuracy'], score['RMSE_Norm'], score['RMSE_Real'], score['MAPE']))

        del model
        tf.keras.backend.clear_session()
    except Exception as e:
        print('+' * 50)
        print('Error {} {} {}'.format(len_closeness, len_period, len_trend))
        print(e)
        print('+' * 50)
        preprocessed_DB_path = os.path.join(os.getcwd(),
                                            'DB/databases_Cleaned/paper_preprocess/{}'.format(DB_name),
                                            'len_closeness_{}_len_period_{}_len_trend_{}'.format(
                                                len_closeness, len_period,
                                                len_trend))
        os.removedirs(preprocessed_DB_path)
        metric = 100
    if metric is None:
        metric = 100
    return metric


def optimize_hyperparameters(fitness, dimensions, nb_iteration, default_parameters, save_result=False, save_result_path=''):
    search_result = skopt.gp_minimize(func=fitness, dimensions=dimensions, n_calls=nb_iteration, x0=default_parameters)
    if save_result:
        skopt.dump(search_result, os.path.join(save_result_path, 'Result'))
    return search_result


def plot_result_search(search_result, dim_name, save_plot=False, save_plot_path=''):
    plot_objective(search_result, dimensions=dim_name)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    if save_plot == True:
        plt.savefig(os.path.join(save_plot_path, 'Objective.jpg'))
    else:
        plt.show()

    plot_evaluations(search_result, dimensions=dim_name)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    if save_plot == True:
        plt.savefig(os.path.join(save_plot_path, 'Evaluations.jpg'))
    else:
        plt.show()
    plot_convergence(search_result)
    if save_plot == True:
        plt.savefig(os.path.join(save_plot_path, 'Convergence.jpg'))
    else:
        plt.show()


def load_search_result(result_path):
    result = skopt.load(os.path.join(result_path, 'Result'))
    return result


def main(search=True, save_csv=True):
    tmp_path = os.path.join(os.getcwd(), 'tmp', DB_name)
    if search:
        default_parameters = [5, 5, 5, 0.005, 2, 2, 32, 'tanh', 0, 0, 0]
        search_result = optimize_hyperparameters(fitness, dimensions, nb_iteration=100,
                                                 default_parameters=default_parameters, save_result=True,
                                                 save_result_path=tmp_path)
    else:
        search_result = load_search_result(tmp_path)
        print('Best parameters are: {}\n Best result: {}'.format(search_result.x, search_result.fun))
        print(sorted(zip(search_result.func_vals, search_result.x_iters)))
        dim_name = ['len_closeness', 'len_period', 'len_trend', 'lr', 'net_type', 'nb_Modules', 'filter_size', 'activation', 'dropout',
                    'l1', 'l2']
        search_result_pd = pd.DataFrame(data=search_result.x_iters, columns=dim_name)
        search_result_pd['RMSE'] = search_result.func_vals
        search_result_pd = search_result_pd.sort_values(by='RMSE')
        if save_csv:
            search_result_pd.to_csv(os.path.join(tmp_path, 'report.csv'), index_label='search_id')
        #plot_result_search(search_result, dim_name, save_plot=False, save_plot_path=tmp_path)
    return search_result

if __name__ == '__main__':
    search_result = main(search=True, save_csv=True)
    dim_name = ['len_closeness', 'len_period', 'len_trend', 'lr', 'net_type', 'nb_Modules', 'filter_size', 'activation',
                'dropout',
                'l1', 'l2']
    #search_result = load_search_result(os.path.join('./tmp', 'BikeNYC'))
    #print(search_result.x_iters)
    #print(search_result.func_vals)
    #plot_evaluations(result=search_result, dimensions=search_result.space.dimensions[3].name)
    #plt.show()
    #print(search_result.x)
    #print(search_result.fun)
    #print(search_result.space)
    #print(search_result.specs)
    #print(search_result.models)
    #plot_objective(search_result)
    #plot_result_search(search_result, dim_name=dim_name)
