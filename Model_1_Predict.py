import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import streamlit as st
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


## Apply Standard Scaler
from sklearn.preprocessing import StandardScaler

## Constants

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
X_test = None
Y_test = None

### Upload test data
upload_X_test = st.file_uploader("upload feature file",type={"csv", "txt"})

if upload_X_test is not None:
    X_test = np.loadtxt(upload_X_test, delimiter=",", dtype=float)
    
    upload_Y_test = st.file_uploader("upload class file",type={"csv", "txt"})
    
    if upload_Y_test is not None:
        Y_test = np.loadtxt(upload_Y_test, delimiter=",", dtype=float)
    


if X_test is not None and Y_test is not None:
    model_1 = pickle.load(open('Model_1.pkl', 'rb'))
    
    y_pred = model_1.predict(X_test)
    
    st.write("Exploring Features")
    st.write(X_test)
    st.write("Exploring Classes")
    st.write(Y_test)

    ###################### Evaluate the performance of the model on the test data ####################
    test_loss, test_acc = model_1.evaluate(X_test, Y_test)
    st.write('Test accuracy:', test_acc)
    st.write('Test Losss:', test_loss)
