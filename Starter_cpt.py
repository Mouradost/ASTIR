import tensorflow as tf
import Preprocessing, Checker
import os
from Networks import ConvLSTM_Inception_ResNet, Trainer
tf.set_random_seed(1337)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nbr_gpu = 1

def main_train(train=True, tmp_name='tmp'):
    DB_names = ['BikeNYC']
    activations = ['tanh']
    filters_size = [16]
    nbs_Modules = [1]
    nets_names = ['ConvLSTM_Inception_ResNet']
    len_closeness = 4
    len_period = 2
    len_trend = 5
    dropout = 0.0
    dropout_inception_block = 0.1
    l1_rec = 0
    l2_rec = 0
    l1_ker = 0
    l2_ker = 0
    l1_act = 0
    l2_act = 0
    use_bn = False
    use_add_bn = False
    epoch = 500
    batch_size = 32
    learning_rate = 0.005
    decay = 0.0
    load_weights = True
    save_weight = True

    for DB_name in DB_names:
        for len_closeness in range(4, 5):
            for len_period in range(0, 6):
                for len_trend in range(1, 2):
                    if DB_name == 'TaxiBJ':
                        c_conf = (len_closeness, 2, 32, 32)
                        p_conf = (len_period, 2, 32, 32)
                        t_conf = (len_trend, 2, 32, 32)
                        output_shape = (2, 32, 32)
                        external_shape = (28,)
                        X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(
                            DB_name,
                            len_closeness,
                            len_period,
                            len_trend)
                        X_train, X_test = Preprocessing.prepare_data_as_a_sequence(X_train, X_test, len_closeness=len_closeness,
                                                                                   len_period=len_period, len_trend=len_trend,
                                                                                   channel=output_shape[0])
                        save_path_or = os.path.join(os.getcwd(), tmp_name, '{}'.format(DB_name))
                        os.makedirs(save_path_or, exist_ok=True)

                    else:
                        c_conf = (len_closeness, 2, 16, 8)
                        p_conf = (len_period, 2, 16, 8)
                        t_conf = (len_trend, 2, 16, 8)
                        output_shape = (2, 16, 8)
                        external_shape = ()
                        X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(
                            DB_name,
                            len_closeness,
                            len_period,
                            len_trend)
                        X_train, X_test = Preprocessing.prepare_data_as_a_sequence(X_train, X_test, len_closeness=len_closeness,
                                                                                   len_period=len_period, len_trend=len_trend,
                                                                                   channel=output_shape[0])
                        save_path_or = os.path.join(os.getcwd(), tmp_name, '{}'.format(DB_name))
                        os.makedirs(save_path_or, exist_ok=True)

                    for net_name in nets_names:
                        net_name_tmp = net_name
                        for activation in activations:
                            for filter_size in filters_size:
                                for nb_Modules in nbs_Modules:
                                    save_path = os.path.join(save_path_or, 'c_{}_p_{}_t_{}'.format(len_closeness, len_period, len_trend))
                                    os.makedirs(save_path, exist_ok=True)
                                    for net_type in range(0, 1):
                                        if net_name == 'ConvLSTM_Inception_ResNet':
                                            net_name = net_name_tmp + '_ASTIR_{}'.format(net_type)
                                            model = ConvLSTM_Inception_ResNet.convLSTM_Inception_ResNet_network(c_conf=c_conf,
                                                                                                                p_conf=p_conf,
                                                                                                                t_conf=t_conf,
                                                                                                                output_shape=output_shape,
                                                                                                                external_shape=external_shape,
                                                                                                                nb_modules=nb_Modules,
                                                                                                                filters=filter_size,
                                                                                                                kernel_size=(3, 3),
                                                                                                                strides=(1, 1),
                                                                                                                padding='same',
                                                                                                                data_format='channels_first',
                                                                                                                activation=activation,
                                                                                                                dropout=dropout,
                                                                                                                dropout_inception_block=dropout_inception_block,
                                                                                                                l1_rec = l1_rec,
                                                                                                                l2_rec = l2_rec,
                                                                                                                l1_ker = l1_ker,
                                                                                                                l2_ker = l2_ker,
                                                                                                                l1_act = l1_act,
                                                                                                                l2_act = l2_act,
                                                                                                                use_bn = use_bn,
                                                                                                                use_add_bn = use_add_bn,
                                                                                                                types=net_type)
                                        print('*' * 50)
                                        print('Tensorflow version : {}'.format(tf.VERSION))
                                        print('Keras version : {}'.format(tf.keras.__version__))
                                        print('_' * 50)
                                        print(
                                            'Database name : {}\nModel type : {}\nActivation : {}\nNumber of layers : {}\nNumber of filters : {}'.format(
                                                DB_name, net_name, activation, nb_Modules, filter_size))
                                        print('_' * 50)
                                        print(
                                            'Closeness shape : {}\nPeriod shape : {}\nTemporal shape : {}\nOutput shape : {}\nExternals shape : {}'.format(
                                                c_conf, p_conf, t_conf, output_shape, external_shape))
                                        print('_' * 50)
                                        print(
                                            'Epoch : {}\nBatch size : {}\nLearning rate : {}\nDecray : {}\nDropout : {}'.format(
                                                epoch, batch_size, learning_rate, decay, dropout))
                                        print('*' * 50)
                                        if nbr_gpu > 1:
                                            try:
                                                model = tf.keras.utils.multi_gpu_model(model, gpus=nbr_gpu)
                                                print("Training using multiple GPUs..")
                                            except Exception as e:
                                                print("Training using single GPU..")
                                                print("Error : ", e)
                                        if train:
                                            info, score = Trainer.train(model=model, X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test,
                                                                        mmn=mmn, DB_name=DB_name,
                                                                        epochs=epoch,
                                                                        batch_size=batch_size, network_name=net_name,
                                                                        nb_modules=nb_Modules,
                                                                        filters=filter_size, activation=activation,
                                                                        learning_rate=learning_rate, decay=decay,
                                                                        load_weights=load_weights,
                                                                        save_weight=save_weight, save_path=save_path)
                                        else:
                                            model.load_weights(os.path.join(save_path,
                                                                            'network_name={},nb_modules={},filters={},activation={}_weights.h5'.format(
                                                                                net_name, nb_Modules, filter_size, activation)))
                                            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay),
                                                          metrics=[Trainer.RMSE, Trainer.RMSE_Real(mmn=mmn, DB_name=DB_name), 'accuracy', 'MAPE'])

                                            Trainer.evaluation(model=model, X_test=X_train, Y_test=Y_train, batch_size=batch_size)
                                            Trainer.evaluation(model=model, X_test=X_test, Y_test=Y_test,
                                                               batch_size=batch_size)
                                            print(len(timestamp_test))
                                            Checker.compare_prediction_ground(model=model, X_test=X_test, Y_test=Y_test,
                                                                              save_img=False, save_img_path='')
                                        del model
                                        tf.keras.backend.clear_session()


if __name__ == '__main__':
   main_train(train=False, tmp_name='tmp_cpt')