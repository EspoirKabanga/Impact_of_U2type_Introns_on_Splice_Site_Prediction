
import tensorflow as tf

def IntSplicer(length):
    model = tf.keras.models.Sequential(name='IntSplicer')
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

    model_SS = tf.keras.models.Sequential(name='SpliceRover')
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

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)

    model_SS.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  

    return model_SS

def splice_finder(length):

    model = tf.keras.models.Sequential(name='SpliceFinder')
        
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def DeepSplicer(length):

    model = tf.keras.models.Sequential(name='DeepSplicer')

    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model