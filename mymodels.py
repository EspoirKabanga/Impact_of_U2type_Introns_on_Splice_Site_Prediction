import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

def IntSplice(length):
    model = tf.keras.models.Sequential()
    layer1 = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=4, padding='same',
                                    batch_input_shape=(None, length, 4), activation='relu')

    layer2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
                                    activation='relu')

    layer3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
                                    activation='relu')
    
    layer4 = tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same',
                                    activation='relu')

    model.add(layer1)
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(layer2)
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(layer3)
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(layer4)
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, activation='relu', name='layer_dense')) # 20
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(2, activation='softmax', name='out'))

    # model.summary()

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  # categorical

    return model


def SpliceRover_Model(Length):

    # ==========Splice Site Model==================================

    model_SS = tf.keras.models.Sequential()
    layer1 = tf.keras.layers.Conv1D(filters=70, kernel_size=9, strides=4, padding='same', batch_input_shape=(None, Length, 4),
                           activation='relu', name='conv1')

    layer2 = tf.keras.layers.Conv1D(filters=100, kernel_size=7, strides=1, padding='same',
                           activation='relu', name='conv2')

    layer3 = tf.keras.layers.Conv1D(filters=100, kernel_size=7, strides=1, padding='same',
                           activation='relu', name='conv3')

    layer4 = tf.keras.layers.Conv1D(filters=200, kernel_size=7, strides=1, padding='same',
                           activation='relu', name='conv4')

    layer5 = tf.keras.layers.Conv1D(filters=250, kernel_size=7, strides=1, padding='same',
                           activation='relu', name='conv5')
    # ---------------------------------------------

    model_SS.add(layer1)
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(layer2)
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(layer3)
    model_SS.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=1))
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(layer4)
    model_SS.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=1))
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(layer5)
    model_SS.add(tf.keras.layers.MaxPool1D(pool_size=4, strides=1))
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(tf.keras.layers.Flatten())

    model_SS.add(tf.keras.layers.Dense(512, activation='relu', name='layer_dense'))
    model_SS.add(tf.keras.layers.Dropout(0.2))

    model_SS.add(tf.keras.layers.Dense(2, activation='softmax', name='out'))

    # model_SS.summary()

    # adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # model_SS.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  # categorical
    model_SS.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.05, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model_SS


def splice_finder(length):

    model = tf.keras.models.Sequential()
        
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    adam = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def DeepSplicer(length):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    adam = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model