from Networks import Trainer, DeepST_Base_Line, ConvLSTM, ConvLSTM_Base_Line, ConvLSTM_ResNet, ConvLSTM_Inception_ResNet
import Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def info_visualisation(DB_name, Net_name, nb_modules, filters, activation, save_path=''):
    if save_path == '':
        save_path = os.path.join(os.getcwd(), 'tmp')
        save_path = os.path.join(save_path, '{}'.format(DB_name))
    info = Trainer.reader(DB_name, Net_name, nb_modules, filters, activation, save_path)
    main_summary = pd.DataFrame(info*100, index=np.arange(info.shape[0]), columns=['Acc', 'Loss'])
    plt.figure(1)
    plt.subplot(211)
    main_summary['Acc'].plot()
    plt.title('Accuracy (%)')
    plt.grid(True)
    plt.subplot(212)
    main_summary['Loss'].plot(color='r')
    plt.title('Loss (%)')
    plt.grid(True)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def compare_prediction_ground(model, X_test, Y_test, save_img=False, save_img_path=''):
    prediction = model.predict(X_test, batch_size=None, verbose=0, steps=None)
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(prediction[0][0])
    plt.title('Inflow_Prediction')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(Y_test[0][0])
    plt.title('Inflow_Ground')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(prediction[0][1])
    plt.title('Outflow_Prediction')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(Y_test[0][1])
    plt.title('Outflow_Ground')
    plt.colorbar()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    if save_img == True:
        plt.savefig(save_img_path+'_Comparison.jpg')
    else:
        plt.show()
    plot_data_summary(prediction, Y_test, type_summary='Mean', flow_type='Inflow')
    plot_data_summary(prediction, Y_test, type_summary='Mean', flow_type='Outflow')



def create_data_summary(data):
    indexes = pd.MultiIndex.from_product([['Min', 'Max', 'Mean'], ['Inflow', 'Outflow']], names=['Measure', 'Type'])
    test = np.hstack([np.min(data, axis=(2, 3)), np.max(data, axis=(2, 3)), np.mean(data, axis=(2, 3))])
    data_summary = pd.DataFrame(test, columns=indexes)
    return data_summary


def plot_data_summary(prediction, ground, type_summary='', flow_type=''):
    prediction_summary = create_data_summary(prediction)
    ground_summary = create_data_summary(ground)
    ax = prediction_summary[type_summary][flow_type].plot(color='b', label='Prediction')
    ground_summary[type_summary][flow_type].plot(ax=ax, mark_right=True, color='r', label='Ground')
    plt.legend()
    plt.title(flow_type)
    plt.show()


def predict(net_name, nb_Modules, filter_size, activation, dropout, data, Y_test, save_img=False, save_img_path='', net_type=None, c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), output_shape=(2, 32, 32), external_shape=(20,), load_weight=True, weight_path=''):
    if net_name == 'DeepST_Base_Line':
        model = DeepST_Base_Line.DeepST_network(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                                                    output_shape=output_shape, external_shape=external_shape,
                                                    nb_modules=nb_Modules, filters=filter_size, kernel_size=(3, 3),
                                                    strides=(1, 1), padding='same', data_format='channels_first',
                                                    activation=activation, dropout=dropout)
    elif net_name == 'ConvLSTM_Base_Line':
        model = ConvLSTM_Base_Line.convLSTM_network(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                                                    output_shape=output_shape, external_shape=external_shape,
                                                    nb_modules=nb_Modules, filters=filter_size, kernel_size=(3, 3),
                                                    strides=(1, 1), padding='same', data_format='channels_first',
                                                    activation=activation, dropout=dropout)
    elif net_name == 'ConvLSTM':
        model = ConvLSTM.convLSTM_network(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                                          output_shape=output_shape, external_shape=external_shape, nb_modules=nb_Modules,
                                          filters=filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                          data_format='channels_first', activation=activation, dropout=dropout)
    elif net_name == 'ConvLSTM_ResNet':
        model = ConvLSTM_ResNet.convLSTM_ResNet_network(c_conf=c_conf, p_conf=p_conf,
                                                        t_conf=t_conf, output_shape=output_shape,
                                                        external_shape=external_shape, nb_modules=nb_Modules,
                                                        filters=filter_size, kernel_size=(3, 3), strides=(1, 1),
                                                        padding='same', data_format='channels_first',
                                                        activation=activation, dropout=dropout, bn_first=False,
                                                        bn_last=True)
    else:
        model = ConvLSTM_Inception_ResNet.convLSTM_Inception_ResNet_network(c_conf=c_conf,
                                                                            p_conf=p_conf,
                                                                            t_conf=t_conf,
                                                                            output_shape=output_shape,
                                                                            external_shape=external_shape, nb_modules=nb_Modules,
                                                                            filters=filter_size, kernel_size=(3, 3),
                                                                            strides=(1, 1), padding='same',
                                                                            data_format='channels_first',
                                                                            activation=activation, dropout=dropout,
                                                                            bn_first=False, bn_last=True,
                                                                            types=net_type)
        #net_name = net_name + '_{}'.format(net_type)
    if load_weight == True:
        print('Load the weights')
        model.load_weights(os.path.join(weight_path,
                                    'network_name={},nb_modules={},filters={},activation={}_weights.h5'.format(
                                        net_name, nb_Modules, filter_size, activation)))
    if save_img:
        save_img_path = os.path.join(save_img_path, 'network_name={},nb_modules={},filters={},activation={}'.format(
                                        net_name, nb_Modules, filter_size, activation))
    compare_prediction_ground(model=model, X_test=data, Y_test=Y_test, save_img=save_img, save_img_path=save_img_path)




if __name__ == '__main__':

    net_name, nb_Modules, filter_size, activation, dropout, net_type = 'ConvLSTM_Inception_ResNet_type_1', 1, 16, 'tanh', 0.0, 1
    len_closeness = 12
    len_period = 1
    len_trend = 12
    weight_path = 'E:/PycharmProjects/First_Paper/tmp_good_result/BikeNYC'
    save_img_path = weight_path
    X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = Preprocessing.get_DB(DB_name='BikeNYC', len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
    X_train, X_test = Preprocessing.prepare_data_as_a_sequence(X_train, X_test, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, channel=2)

    #info_visualisation(DB_name='BikeNYC', Net_name=net_name, nb_modules=nb_Modules, filters=filter_size, activation=activation, save_path='E:/PycharmProjects/First_Paper/tmp/BikeNYC')
    predict(net_name, nb_Modules, filter_size, activation, dropout, X_test, Y_test=Y_test, net_type=net_type, c_conf=(len_closeness, 2, 16, 8),
                                                                                p_conf=(len_period, 2, 16, 8),
                                                                                t_conf=(len_trend, 2, 16, 8),
                                                                                output_shape=(2, 16, 8), external_shape=(), weight_path=weight_path, save_img=False, save_img_path=save_img_path)
