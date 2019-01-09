import tensorflow as tf
tf.set_random_seed(1337)

def attention(inputs, data_format='channels_first', filters=32):
    x = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       data_format=data_format, return_sequences=True)(inputs)
    x = tf.keras.layers.MaxPooling3D(data_format=data_format)(x)
    x = tf.keras.layers.ConvLSTM2D(filters=filters // 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       data_format=data_format, return_sequences=True)(x)
    x = tf.keras.layers.UpSampling3D(data_format=data_format)(x)
    inputs_conv = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       data_format=data_format, return_sequences=True)(inputs)
    x = tf.keras.layers.add([x, inputs_conv])
    result = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                       data_format=data_format, return_sequences=True)(x)
    return result


def squeeze_excite_block(input, len_seq, channel, map_height, map_width, data_format='channels_first', ratio=2, layer_count=0):
    with tf.keras.backend.name_scope('Squeeze_Block_{}'.format(layer_count)):
        init = tf.keras.layers.Reshape((len_seq*channel, map_height, map_width))(input)
        channel_axis = 1 if data_format == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(init)
        se = tf.keras.layers.Reshape(se_shape)(se)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if data_format == 'channels_first':
            se = tf.keras.layers.Permute((3, 1, 2))(se)

        x = tf.keras.layers.multiply([init, se])
        x = tf.keras.layers.Reshape((len_seq, channel, map_height, map_width))(x)
    return x

def convLSTM_block(inputs, filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', data_format='channels_first', activation='tanh', dropout=0.0, l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, return_sequences=True, use_bn=False):
    x = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   data_format=data_format, return_sequences=return_sequences,
                                   recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rec, l2=l2_rec),
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_ker, l2=l2_ker),
                                   activity_regularizer=tf.keras.regularizers.l1_l2(l1=l1_act, l2=l2_act))(inputs)
    if use_bn:
        if data_format == 'channels_first':
            if return_sequences:
                channel_axes = 2
            else:
                channel_axes = 1
        else:
            channel_axes = -1
        x = tf.keras.layers.BatchNormalization(axis=channel_axes, scale=False)(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x

def convLSTM_Inception_ResNet_module(inputs, layer_count, filters=32, strides=(1, 1), padding='same', data_format='channels_first', activation='tanh', types=1, dropout=0.0, l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, use_bn=False, use_add_bn=True):
    with tf.keras.backend.name_scope('Inception_ResNet_ConvLSTM_Block_{}'.format(layer_count)):
        if types == 0:
            a = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(1, 3), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(3, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            b = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            d = tf.keras.layers.concatenate([a, b], axis=2)
            x = convLSTM_block(d, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)


        elif types == 1:
            a =convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(1, 7), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(7, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            b = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            d = tf.keras.layers.concatenate([a, b], axis=2)
            x = convLSTM_block(d, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        else:
            a = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            a = convLSTM_block(a, filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            b = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            b = convLSTM_block(b, filters=filters, kernel_size=(3, 3), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            c = convLSTM_block(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

            d = tf.keras.layers.concatenate([a, b, c], axis=2)
            x = convLSTM_block(d, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn)

        x = tf.keras.layers.add([x, inputs])
        if use_add_bn:
            if data_format == 'channels_first':
                channel_axes = 2
            else:
                channel_axes = -1
            x = tf.keras.layers.BatchNormalization(axis=channel_axes, scale=False)(x)
    return x



def convLSTM_Inception_ResNet_network(
    c_conf=(3, 1, 10, 10), p_conf=(3, 1, 10, 10), t_conf=(3, 1, 10, 10), output_shape=(1, 10, 10), external_shape=(23,),
    nb_modules=2,
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format='channels_first',
    activation='tanh',
    dropout=0.2,
    dropout_inception_block=0,
    l1_rec=0, l2_rec=0, l1_ker=0, l2_ker=0, l1_act=0, l2_act=0, use_bn=False, use_add_bn=True,
    types=0
    ):
    inputs, outputs = [], []
    if c_conf[0] > 0 and p_conf[0] > 0 and t_conf[0] > 0:
        inputs_shape = [c_conf, p_conf, t_conf]
    if c_conf[0] == 0 and p_conf[0] > 0 and t_conf[0] > 0:
        inputs_shape = [p_conf, t_conf]
    if c_conf[0] > 0 and p_conf[0] == 0 and t_conf[0] > 0:
        inputs_shape = [c_conf, t_conf]
    if c_conf[0] > 0 and p_conf[0] > 0 and t_conf[0] == 0:
        inputs_shape = [c_conf, p_conf]
    for input_shape in inputs_shape:
        len_seq, channel, map_height, map_width = input_shape
        input_img = tf.keras.layers.Input(shape=(len_seq, channel, map_height, map_width))
        inputs.append(input_img)
        x = convLSTM_block(input_img, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=False)
        for i in range(nb_modules):
            x = convLSTM_Inception_ResNet_module(x, layer_count=i, filters=filters, strides=strides, padding=padding, dropout=dropout_inception_block,
                                                 data_format=data_format, activation=activation, types=types, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, use_bn=use_bn, use_add_bn=True)
            x = squeeze_excite_block(input=x, len_seq=len_seq, channel=filters, map_height=map_height, map_width=map_width, data_format=data_format, ratio=2, layer_count=i)
            #x = attention(x, data_format=data_format, filters=filters)
        x = convLSTM_block(x, filters=channel, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, dropout=dropout, l1_rec=l1_rec, l2_rec=l2_rec, l1_ker=l1_ker, l2_ker=l2_ker, l1_act=l1_act, l2_act=l2_act, return_sequences=False, use_bn=False)


        outputs.append(x)
    added = tf.keras.layers.add(outputs)
    if len(external_shape) != None and len(external_shape) > 0:
        external = tf.keras.layers.Input(shape=external_shape)
        inputs.append(external)
        y = tf.keras.layers.Dense(10, activation=activation)(external)
        y = tf.keras.layers.Dense(output_shape[0] * output_shape[1] * output_shape[2], activation=activation)(y)
        y = tf.keras.layers.Reshape(output_shape)(y)
        added = tf.keras.layers.add([added, y])
    if use_add_bn:
        if data_format == 'channels_first':
            channel_axes = 1
        else:
            channel_axes = -1
        added = tf.keras.layers.BatchNormalization(axis=channel_axes, scale=False)(added)
    result = tf.keras.layers.Activation(activation='tanh')(added)
    model = tf.keras.models.Model(inputs=inputs, outputs=result)

    print(model.summary())

    return model

if __name__ == '__main__':
    for types in range(3):
        convLSTM_Inception_ResNet_network(types=types)
