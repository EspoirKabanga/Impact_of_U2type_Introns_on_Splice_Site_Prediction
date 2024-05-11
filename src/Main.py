# Pleae change the all the paths before running the code

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import random

# Set seed for reproducibility
seed_value = 42
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import mymodels as mdl
import mycallbacks as clbk
import savemodel as svmdl
import datapreprocess as prc
from keras import utils
from keras import backend as K
import glob
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score
from math import *
import csv

#########################################################################################################################
def save_metrics_to_csv(model_name, batch_size, fold_metrics_list, avg_metrics, std_metrics, data_category):
    # Create a CSV file with the model name and batch size in the filename
    filename = f"RDM_new_results/yyy_{model_name}_{data_category}_don_{batch_size}.csv"

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow(['Fold', 'Sensitivity', 'Precision', 'F1-score', 'False Positive Rate',
                            'False Discovery Rate', 'Specificity', 'Recall', 'MCC'])

        # Write fold metrics
        for fold_idx, fold_metrics in enumerate(fold_metrics_list, start=1):
            rec, pr, f1, fpr, fdr, sp, sn, mcc = fold_metrics
            csv_writer.writerow([f'fold {fold_idx}', f'{rec:.4f}', f'{pr:.4f}', f'{f1:.4f}', f'{fpr:.4f}',
                                f'{fdr:.4f}', f'{sp:.4f}', f'{sn:.4f}', f'{mcc:.4f}'])

        # Write average and standard deviation metrics
        csv_writer.writerow(['Average', f'{avg_metrics[0]:.4f}', f'{avg_metrics[1]:.4f}', f'{avg_metrics[2]:.4f}',
                            f'{avg_metrics[3]:.4f}', f'{avg_metrics[4]:.4f}', f'{avg_metrics[5]:.4f}',
                            f'{avg_metrics[6]:.4f}', f'{avg_metrics[7]:.4f}'])
        csv_writer.writerow(['Std Deviation', f'{std_metrics[0]:.4f}', f'{std_metrics[1]:.4f}',
                            f'{std_metrics[2]:.4f}', f'{std_metrics[3]:.4f}', f'{std_metrics[4]:.4f}',
                            f'{std_metrics[5]:.4f}', f'{std_metrics[6]:.4f}', f'{std_metrics[7]:.4f}'])

def calculateMetrics(preds, labs):
    
    count = 1
    
    tp, tn, fn, fp = 0, 0, 0, 0
    precision_scores = []
    recall_scores = []

    for (_, p), (_, l) in zip(preds, labs):
        if p >= 0.5 and l == 1:
            tp += 1
        elif p < 0.5 and l == 1:
            fn += 1
        elif p >= 0.5 and l == 0:
            fp += 1
        else:
            tn += 1

        # Calculate precision and recall at each step
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)

    r = tp / (tp + fn) if tp + fn > 0 else 0.0  # Recall
    p = tp / (tp + fp) if tp + fp > 0 else 0.0  # Precision

    f1 = 2 * r * p / (r + p) if r + p > 0 else 0.0  # F1-Score

    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0  # False Positive Rate

    fdr = fp / (tp + fp) if tp + fp > 0 else 0.0  # False Discovery Rate

    sp = tn / (tn + fp) if tn + fp > 0 else 0.0  # Specificity

    sn = tp / (tp + fn) if tp + fn > 0 else 0.0  # Sensitivity

    mcc = ((tp * tn) - (fp * fn)) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0.0  # Matthews Correlation Coefficient

    return (r, p, f1, fpr, fdr, sp, sn, mcc)


dataset = ['long', 'short', 'mix_short_long', 'single', 'multiple', 'mix_single_multiple']

for data_category in dataset:

    if data_category == 'long' or data_category == 'short' or data_category == 'mix_short_long':
        n_sample = 3212
    else:
        n_sample = 2377

    # data_category = 'multiple'
    data_pos, data_neg = prc.read_file(f'{data_category}_donor.pos', 'dataset/donors.neg')  # Replace 'donor' by 'acceptor' for other experiments
    data_pos = random.sample(data_pos, n_sample)  # Not to use for mixed data

    data_neg = random.sample(data_neg, len(data_pos) * 5)  # Taking pos:neg ratio = 1:5

    print('----------------------------------------')
    print('initial pos size = ', len(data_pos))
    print('initial neg size = ', len(data_neg))
    print('----------------------------------------')

    train_X, train_y = np.array(data_pos + data_neg), np.array([1] * len(data_pos) + [0] * len(data_neg))

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)

    ################### Reshaping the test set ####################
    X_test, y_test = test_X.reshape(-1, 402, 4), utils.to_categorical(test_y, num_classes=2)
    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    ####################################################################################


    batches = [64] # Tried 32, 128 and 256 for IntSplicer Model

    for bt in batches:

        mdls = [mdl.DeepSplicer(402), mdl.splice_finder(402), mdl.IntSplicer(402), mdl.SpliceRover_Model(402)]

        for md in mdls:

            fold_metrics_list = []


            # Initialize 5-fold cross-validation
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            fold = 1

            # Loop through the k-folds
            for train_index, valid_index in kf.split(train_X, train_y):
                
                # exist or not.
                if not os.path.exists(f"/home/ekabanga/Plot_CB/best_weights_dn{data_category}"):
                    # then create it.
                    os.makedirs(f"/home/ekabanga/Plot_CB/best_weights_dn{data_category}")
                        
                prj_name = f'{md.name}_zzz_{data_category}_don_{str(bt)}_fold_{str(fold)}'

                model_callbacks = clbk.my_callbacks(prj_name, data_category)

                # train and valid
                X_train, X_valid = train_X[train_index], train_X[valid_index]
                y_train, y_valid = train_y[train_index], train_y[valid_index]

                #######################################################
                print('size of train = ', len(X_train))
                print('size of valid = ', len(X_valid))
                print('size of test = ', len(X_test))
                print('--------------------------------------')
                #######################################################

                X_train, y_train = X_train.reshape(-1, 402, 4), utils.to_categorical(y_train, num_classes=2)
                X_valid, y_valid = X_valid.reshape(-1, 402, 4), utils.to_categorical(y_valid, num_classes=2)

                X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
                X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)

                print(f'training fold {fold} for batch {bt} for {md.name} for {data_category}')
                history = md.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=bt, callbacks=model_callbacks, verbose=2)
                list_of_files = glob.glob(f'/home/ekabanga/Plot_CB/best_weights_dn{data_category}'+'/*')

                latest_file = max(list_of_files, key=os.path.getctime)

                md.load_weights(latest_file)

                #################################################################################################################

                loss, accuracy = md.evaluate(X_test, y_test)
                pred = md.predict(X_test)

                fold_metrics = calculateMetrics(pred, y_test)
                fold_metrics_list.append(fold_metrics)

                svmdl.save_model(md, history, prj_name, data_category) # Saving the model

                print('#################################################################################')
                fold += 1

                # Clear session to free up GPU memory
                K.clear_session()
                # Deleting the model best weight directory
                shutil.rmtree(f'/home/ekabanga/Plot_CB/best_weights_dn{data_category}')

            # Print metrics summary for the current model and batch size
            fold_metrics_array = np.array(fold_metrics_list)
            avg_metrics = np.mean(fold_metrics_array, axis=0)
            std_metrics = np.std(fold_metrics_array, axis=0)

            # Save metrics to CSV file
            save_metrics_to_csv(md.name, bt, fold_metrics_list, avg_metrics, std_metrics, data_category)