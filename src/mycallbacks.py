import tensorflow as tf
import os

# exist or not.
if not os.path.exists("/home/ekabanga/Plot_CB/best_weights"):

    # then create it.
    os.makedirs("/home/ekabanga/Plot_CB/best_weights")

def my_callbacks(project_name=None):
    
    checkpoint_filepath = 'best_weights/' + project_name + '_weights.{epoch:02d}-{val_loss:.2f}.h5'  # Recording the validation loss, so later to load weights from model with lowest valid loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                                                   mode='min', save_freq='epoch', save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5, restore_best_weights=True)

    # reduce_LR_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
    #                                                           mode='min') # min_lr=0.001'''

    my_callback = [model_checkpoint_callback, early_stopping_callback] # , reduce_LR_callback

    return my_callback