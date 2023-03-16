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
from sklearn.metrics import precision_score , accuracy_score , f1_score,recall_score
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical
from sklearn.metrics import classification_report


st.set_option('deprecation.showPyplotGlobalUse', False)
X_test = None
y_test = None

### Upload test data
upload_X_test = st.file_uploader("upload feature file",type={"csv", "txt"})

if upload_X_test is not None:
    X_test = np.loadtxt(upload_X_test, delimiter=",", dtype=float)
    
    upload_Y_test = st.file_uploader("upload class file",type={"csv", "txt"})
    
    if upload_Y_test is not None:
        y_test = np.loadtxt(upload_Y_test, delimiter=",", dtype=float)
    


if X_test is not None and y_test is not None:
    model_2 = pickle.load(open('Model_2.pkl', 'rb'))
    
    # Evaluating the model (1D Convolutional)
    loss, accuracy= model_2.evaluate(X_test, to_categorical (y_test), verbose=0)

    y_pred = model_2.predict(X_test)
    y_pred = (y_pred > 0.5).astype(np.int)
    y_test = to_categorical(y_test)
    y_test = y_test.reshape(-1)
    y_pred = np.round(y_pred).reshape(-1)
    f1_score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.write(" 1D Convolutional Loss on test set:", loss)
    st.write(" 1D Convolutional Accuracy on test set:", accuracy)

    st.write(" 1D Convolutional F1_score on test set:", f1_score)
    st.write(" 1D Convolutional Precision on test set:", precision)
    st.write(" 1D Convolutional Recall on test set:", recall)


    # # Evaluating the model (1D Convolutional)
    # loss, accuracy= model_2.evaluate(X_test, to_categorical (y_test), verbose=0)

    # print(" 1D Convolutional Loss on test set:", loss)
    # print(" 1D Convolutional Accuracy on test set:", accuracy)

    cm = confusion_matrix(y_test, y_pred) 
    
    st.write("Confusion matrix:")
    st.write(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.xaxis.set_ticklabels(['Class 0', 'Class 1'])
    ax.yaxis.set_ticklabels(['Class 0', 'Class 1'])

    # Loop over data dimensions and create text annotations
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

    # Display the plot in Streamlit
    st.pyplot(fig)



    import seaborn as sns
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot()
