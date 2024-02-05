import mymodels as mdl
import mycallbacks as clbk
import savemodel as svmdl
import datapreprocess as prc
from tensorflow.keras import utils
import glob
import os
import numpy as np
from sklearn.model_selection import KFold


batches = [32, 64, 128, 256, 512]

for bt in batches:

    mdls = [mdl.DeepSplicer(402), mdl.splice_finder(402), mdl.SpliceRover_Model(402), mdl.IntSplice(402)]

    for md in range(len(mdls)):
        if md == 0:
            mdl_name = 'DeepSplicer'
        elif md == 1:
            mdl_name = 'SpliceFinder'
        elif md == 2:
            mdl_name = 'SpliceRover'
        else:
            mdl_name = 'IntSplice'

        # Load positive and negative sequences
        data_pos, data_neg = prc.read_file('dataset/unmutated/TAIR_don.pos', 'dataset/unmutated/TAIR_don.neg')

        # Initialize 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold = 1

        # Loop through the k-folds
        for (train_pos_index, test_valid_pos_index), (train_neg_index, test_valid_neg_index) in zip(kf.split(data_pos),
                                                                                                    kf.split(data_neg)):
            prj_name = f'{mdl_name}_unmutated_don_{str(bt)}_fold_{str(fold)}'

            model_callbacks = clbk.my_callbacks(prj_name)

            # Splitting test and validation datasets
            half_split_pos = len(test_valid_pos_index) // 2
            valid_pos_index, test_pos_index = test_valid_pos_index[half_split_pos:], test_valid_pos_index[:half_split_pos]

            half_split_neg = len(test_valid_neg_index) // 2
            valid_neg_index, test_neg_index = test_valid_neg_index[half_split_neg:], test_valid_neg_index[:half_split_neg]

            # Extracting sequences based on calculated indices
            train_pos_seqs, valid_pos_seqs, test_pos_seqs = [data_pos[i] for i in train_pos_index], [data_pos[i] for i in
                                                                                                    valid_pos_index], [
                                                                data_pos[i] for i in test_pos_index]
            train_neg_seqs, valid_neg_seqs, test_neg_seqs = [data_neg[i] for i in train_neg_index], [data_neg[i] for i in
                                                                                                    valid_neg_index], [
                                                                data_neg[i] for i in test_neg_index]

            # Preparing datasets for training, validation and testing
            train_X, train_Y = np.array(train_pos_seqs + train_neg_seqs), np.array(
                [1] * len(train_pos_seqs) + [0] * len(train_neg_seqs))
            valid_X, valid_Y = np.array(valid_pos_seqs + valid_neg_seqs), np.array(
                [1] * len(valid_pos_seqs) + [0] * len(valid_neg_seqs))
            test_X, test_Y = np.array(test_pos_seqs + test_neg_seqs), np.array(
                [1] * len(test_pos_seqs) + [0] * len(test_neg_seqs))

            # Reshape data for CNN and one-hot encode labels
            train_X, train_Y = train_X.reshape(-1, 402, 4), utils.to_categorical(train_Y, num_classes=2)
            valid_X, valid_Y = valid_X.reshape(-1, 402, 4), utils.to_categorical(valid_Y, num_classes=2)
            test_X, test_Y = test_X.reshape(-1, 402, 4), utils.to_categorical(test_Y, num_classes=2)

            train_X, train_Y = train_X.astype(np.float32), train_Y.astype(np.float32)
            test_X, test_Y = test_X.astype(np.float32), test_Y.astype(np.float32)
            valid_X, valid_Y = valid_X.astype(np.float32), valid_Y.astype(np.float32)

            # Build and train the CNN model
            model = mdls[md]

            model.summary()

            print('training fold ', fold)
            history = model.fit(train_X, train_Y, epochs=30, validation_data=(valid_X, valid_Y), batch_size=bt, callbacks=model_callbacks, verbose=2)
            list_of_files = glob.glob('/home/ekabanga/Plot_CB/best_weights'+'/*')

            latest_file = max(list_of_files, key=os.path.getctime)

            model.load_weights(latest_file)

            svmdl.save_model(model, history, prj_name) # Saving the model

            print('#################################################################################')
            fold += 1