import tensorflow as tf
import numpy as np
import os
import time
import math
from datetime import timedelta

np.random.seed(1337)
tf.set_random_seed(1337)

def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))

def RMSE_Real(mmn, DB_name):
    if DB_name == 'BikeNY':
        map_height, map_width = 16, 8
        nb_area = 81
        m_factor = math.sqrt(1. * map_height * map_width / nb_area)
        def RMSE_real(y_true, y_pred):
            return tf.keras.backend.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))*(mmn._max - mmn._min) / 2. * m_factor
    else:
        def RMSE_real(y_true, y_pred):
            return (tf.keras.backend.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred))*(mmn._max - mmn._min)) / 2.
    return RMSE_real

def train(model, X, Y, X_test, Y_test, mmn, DB_name, epochs=10, batch_size=32, network_name='', nb_modules=1, filters=32, activation='tanh', opt=tf.keras.optimizers.Adam, learning_rate=0.005, decay=0.0, load_weights=True, save_weight=True, save_path='/Users/mac/PycharmProjects/First_Paper', use_dataset=False):
    start = time.time()
    patience = 20 #best between 15 and 20
    if load_weights and os.path.isfile(os.path.join(save_path,
                                                    'network_name={},nb_modules={},filters={},activation={}_weights.h5'.format(network_name, nb_modules, filters, activation))):
        print('Load the weights')
        model.load_weights(os.path.join(save_path,
                                        'network_name={},nb_modules={},filters={},activation={}_weights.h5'.format(network_name, nb_modules, filters, activation)))
    else:
        print('No weights found')
    model.compile(loss='mse', optimizer=opt(lr=learning_rate, decay=decay), metrics=[RMSE, RMSE_Real(mmn=mmn, DB_name=DB_name), 'accuracy', 'MAPE'])
    print('*-*'*20)
    print('The metrics recorded are:\n', model.metrics_names)
    print('*-*' * 20)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_RMSE', factor=0.5, patience=epochs // patience, min_lr=1e-6, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_RMSE', mode='min', patience=patience, min_delta=1e-2, baseline=10)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'network_name={},nb_modules={},filters={},activation={}_logs'.format(network_name, nb_modules, filters, activation)),
        histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False)
    if use_dataset:
        if DB_name == 'TaxiBJ':
            dataset = tf.data.Dataset.from_tensor_slices(({"input_1": X[0], "input_2": X[1], "input_3": X[2], "input_4": X[3]}, Y))
            dataset = dataset.batch(batch_size=batch_size).repeat()
            val_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": X_test[0], "input_2": X_test[1], "input_3": X_test[2], "input_4": X_test[3]}, Y_test))
            val_dataset = val_dataset.batch(batch_size=batch_size).repeat()
        elif DB_name == 'BikeNYC':
            dataset = tf.data.Dataset.from_tensor_slices(({"input_1": X[0], "input_2": X[1], "input_3": X[2]}, Y))
            dataset = dataset.batch(batch_size=batch_size).repeat()
            val_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": X_test[0], "input_2": X_test[1], "input_3": X_test[2]}, Y_test))
            val_dataset = val_dataset.batch(batch_size=batch_size).repeat()
        else:
            dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            dataset = dataset.batch(batch_size=batch_size).repeat()
            val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
            val_dataset = val_dataset.batch(batch_size=batch_size).repeat()
        print('-'*20, '> Step per epoch for train: {}'.format(len(Y)//batch_size))
        print('-' * 20, '> Step per epoch for test: {}'.format(len(Y_test) // batch_size))
        info = model.fit(dataset, epochs=epochs, steps_per_epoch=len(Y)//batch_size, validation_data=val_dataset, shuffle=True, validation_steps=len(Y_test)//batch_size, verbose=1, callbacks=[reduce_lr, tensorboard])
    else:
        info = model.fit(x=X, y=Y, epochs=epochs, validation_split=0.1, shuffle=True, verbose=1, batch_size=batch_size, callbacks=[reduce_lr, tensorboard])

    '''info = model.fit(
        X, Y,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[reduce_lr, early_stopping, tensorboard])'''
    if save_weight:
        # save the weight
        model.save_weights(os.path.join(save_path,
                                        'network_name={},nb_modules={},filters={},activation={}_weights.h5'.format(network_name, nb_modules, filters, activation)))

    score = evaluation(model=model, X_test=X, Y_test=Y, batch_size=batch_size)
    score = evaluation(model=model, X_test=X_test, Y_test=Y_test, batch_size=batch_size)
    print('compilation time : ', str(timedelta(seconds=int(round(time.time() - start)))))
    #saver(info=info, DB_name=DB_name, Net_name=network_name, nb_modules=nb_modules, filters=filters, activation=activation, save_path=save_path)
    return info, score

def evaluation (model, X_test, Y_test, batch_size):
    #score = model.evaluate(x=X_test, y=Y_test, batch_size=len(Y_test)//batch_size, verbose=0)
    score = model.evaluate(x=X_test, y=Y_test, batch_size=len(Y_test) // 48, verbose=0)
    scores = {
        'MSE_Train': score[0],
        'RMSE_Norm': score[1],
        'RMSE_Real': score[2],
        'Accuracy': score[3],
        'MAPE': score[4]
    }
    print('-*-' * 10)
    print('Train score MSE: {:.6f}\nTrain score Accuracy: {:.2%}\nRMSE (norm): {:.6f}\nRMSE (real): {:.4f}\n MAPE: {:.2f}%'.format(
        scores['MSE_Train'], scores['Accuracy'], scores['RMSE_Norm'], scores['RMSE_Real'], scores['MAPE']))
    print('-*-' * 10)
    return scores


def saver(info, DB_name, Net_name, nb_modules, filters, activation, save_path):
    print('****Saving the information for the {} training on {}****'.format(Net_name, DB_name))
    info = np.stack((info.history['acc'], info.history['RMSE']), axis=-1)
    np.save(os.path.join(save_path, 'main_information_{}_{}_{}_{}_{}'.format(DB_name, Net_name, nb_modules, filters, activation)), info)


def reader(DB_name, Net_name, nb_modules, filters, activation, save_path):
    print('****Reading the information for the {} training on {}****'.format(Net_name, DB_name))
    info = np.load(os.path.join(save_path, 'main_information_{}_{}_{}_{}_{}.npy'.format(DB_name, Net_name, nb_modules, filters, activation)))
    print('Data shape : {}'.format(info.shape))
    return info