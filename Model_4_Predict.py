import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import streamlit as st 
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


st.set_option('deprecation.showPyplotGlobalUse', False)

X_test = None
y_test = None

LABELS = ["Normal", "Fraud"]

### Upload test data
upload_X_test = st.file_uploader("upload feature file",type={"csv", "txt"})

if upload_X_test is not None:
    X_test = np.loadtxt(upload_X_test, delimiter=",", dtype=float)
    
    upload_Y_test = st.file_uploader("upload class file",type={"csv", "txt"})
    
    if upload_Y_test is not None:
        y_test = np.loadtxt(upload_Y_test, delimiter=",", dtype=float)
    


if X_test is not None and y_test is not None:
    model_4 = pickle.load(open('model_4.pkl', 'rb'))

    predictions = model_4.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 1), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    
    st.write(error_df.describe())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
    
    st.write('Reconstruction error without fraud')
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fraud_error_df = error_df[error_df['true_class'] == 1]
    
    st.write('Reconstruction error with fraud')
    _ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)

    fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
    roc_auc = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot()

    plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plt.title('Precision for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    st.pyplot()

    plt.plot(recall, precision, 'b', label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    st.pyplot()

    plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
    plt.title('Recall for different threshold values')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Recall')
    st.pyplot()

    # we'll calculate the reconstruction error from the transaction data . If the error is larger than a predefined threshold,
    #  we'll mark it as a fraud:

    threshold = 1.5

    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Fraud" if name == 1 else "Normal")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    st.pyplot()

    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    st.pyplot()


