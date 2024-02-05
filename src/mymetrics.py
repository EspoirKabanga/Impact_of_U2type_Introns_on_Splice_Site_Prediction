import csv
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

import matplotlib
matplotlib.use('WebAgg')
  
# checking if the directory demo_folder 
# exist or not.
if not os.path.exists("/home/ekabanga/Plot_CB/pred_sequence_csv"):

    # if the demo_folder directory is not present 
    # then create it.
    os.makedirs("/home/ekabanga/Plot_CB/pred_sequence_csv")

def calculateMetrics(preds, labs, task_name):

    tp_indx = []
    tn_indx = []
    fn_indx = []
    fp_indx = []

    rows = [['TP','FN','FP','TN']]

    count = 1

    tp, tn, fn, fp = 0, 0, 0, 0
    for (_, p), (_, l) in zip(preds, labs):
        if p >= .5 and l == 1:
            tp += 1
            # tp_indx.append(count)
            rows.append([count, '-','-','-'])
            count += 1
        elif p < .5 and l == 1:
            fn += 1
            # fn_indx.append(count)
            rows.append(['-',count,'-','-'])
            count += 1
        elif p >= .5 and l == 0:
            fp += 1
            # fp_indx.append(count)
            rows.append(['-','-',count,'-'])
            count += 1
        else:
            tn += 1
            # tn_indx.append(count)
            rows.append(['-','-','-',count])
            count += 1

    r = tp / (tp + fn) # recall
    print(r)

    p = tp / (tp + fp) # precision
    print(p)

    f1 = 2 * float(r) * float(p) / (float(r) + float(p)) # f1-score

    fpr = fp / (fp + tn) # false positive rate

    fdr = fp / (tp + fp) # false discovery rate

    sp = tn / (tn + fp) # specificity

    sn = tp / (tp + fn)  # sensitivity

    mcc = ((tp * tn) - (fp * fn)) / sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

    ####### RECORDING THE SEQUENCES AS PER TP/TN,FP,FN INTO A CSV FILE ############

    # header = ["TP", "TN", "FN", "FP"]

    # rows = zip_longest(header, tp_indx, tn_indx, fn_indx, fp_indx)

    # with open('csv_sequence_pred/'+task_name+'.csv', 'w') as f:
    #     csv.writer(f).writerows(rows)

    file = open('pred_sequence_csv/'+task_name+'.csv', 'w', newline ='')
 
    with file:
        writer = csv.writer(file)
        
        for row in rows:
            writer.writerow(row)
        # writing data row-wise into the csv file

    return sp, sn, f1, mcc

def generateAUCROC(preds, labs, task_name):
    #for (_, p), (_, l) in zip(preds, labs):
    fpr, tpr, _ = roc_curve(labs[:, 0], preds[:, 0])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('ROC_'+task_name+'.png')
    plt.close()

def generatePrecisionRecallCurve(preds, labs, task_name):
    precision, recall, _ = precision_recall_curve(labs.ravel(), preds.ravel())
    average_precision = average_precision_score(labs.ravel(), preds.ravel())
    
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve (AP = %0.2f)' % average_precision)
    plt.savefig('Precision_Recall_'+task_name+'.png')
    plt.close()

def generateConfusionMatrix(preds, labs, task_name):
    preds_binary = [1 if p >= 0.5 else 0 for p in preds[:, 0]]
    cm = confusion_matrix(labs[:, 0], preds_binary)
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['True 0', 'True 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('Confusion_Matrix_'+task_name+'.png')
    plt.close()

def generateAUCROC_2(preds, labs, task_name):
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labs, preds)
    roc_auc = auc(fpr, tpr)
    
    # Find the index of the threshold closest to 0.5
    threshold_idx = np.argmin(np.abs(_ - 0.5))

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) at Threshold = 0.5')
    plt.legend(loc="lower right")
    
    # Mark the point at the threshold of 0.5
    plt.scatter(fpr[threshold_idx], tpr[threshold_idx], c='red', marker='o', label='Threshold = 0.5')
    
    plt.savefig('ROC_'+task_name+'.png')
    plt.close()

from sklearn.metrics import PrecisionRecallDisplay
def PrecRecDisplay(X_test, y_test, model):
    display = PrecisionRecallDisplay.from_estimator(
     model, X_test, y_test, name="LinearSVC", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")