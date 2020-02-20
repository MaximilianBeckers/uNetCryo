from tensorflow import keras


def uNet(pretrained_weights = None, input_size = (None, None, None, 1)):

    inputs = keras.layers.Input(shape= input_size)
    conv1 = keras.layers.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = keras.layers.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = keras.layers.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = keras.layers.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = keras.layers.Dropout(0.5)(conv4)

#---------------
    pool4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = keras.layers.Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.pool4)
    conv5 = keras.layers.Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = keras.layers.Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv5))
    merge6 = keras.layers.concatenate([conv4,up6], axis = 4)
    conv6 = keras.layers.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = keras.layers.Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

#---------------

    up7 = keras.layers.Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv6))
    merge7 = keras.layers.concatenate([conv3,up7], axis = 4)
    conv7 = keras.layers.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = keras.layers.Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv7))
    merge8 = keras.layers.concatenate([conv2,up8], axis = 4)
    conv8 = keras.layers.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = keras.layers.Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv8))
    merge9 = keras.layers.concatenate([conv1,up9], axis = 4)
    conv9 = keras.layers.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = keras.layers.Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = keras.layers.Conv3D(1, 1, activation = 'linear')(conv9)

    model = keras.models.Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = keras.optimizers.Adam(lr = 1e-4), loss = 'mean_squared_error')

    return model


