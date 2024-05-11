
import tensorflow as tf
import os


def my_callbacks(model_name=None, data_category=None):
    
    checkpoint_filepath = f'best_weights_dn{data_category}/' + model_name + '_weights.{epoch:02d}-{val_loss:.2f}.h5'  # Recording the validation loss, so later to load weights from model with lowest valid loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                                                   mode='min', save_freq='epoch', save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5, restore_best_weights=True)

    my_callback = [model_checkpoint_callback, early_stopping_callback] 

    return my_callback