import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization , MaxPool1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
from keras.utils import to_categorical
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import precision_score , accuracy_score , f1_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
#Loading the data

data = pd.read_csv('creditcard.csv')
st.write('data.head()')
st.write(data.head())

#check for missing values
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_title('Data plot')
st.pyplot(fig)

import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class_0 = data.loc[data['Class'] == 0]["Time (second)"]
class_1 = data.loc[data['Class'] == 1]["Time (second)"]

hist_data = [class_0, class_1]
group_labels = ['Non-Fradulent', 'Fraudulent']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
st.plotly_chart(fig)

#to check for skewness of data
# visualization of transactions w.r.t time


from pylab import rcParams

rcParams['figure.figsize'] = 18, 7

# Set date column as index
data.set_index(data['Time (second)'], inplace=True)

# Create plot
plt.plot(data.index, data['Amount'])

# Add labels and title
plt.xlabel('Time(in seconds)')
plt.ylabel('Amount')
plt.title('Ditribution of Transaction Time')

# Plot the time series
fig = px.line(data, x=data.index, y='Amount', title='Credit Card Fraud Time Series')
fig.update_layout(xaxis_title='Time', yaxis_title='Transaction Amount')

# Show the plot
st.plotly_chart(fig)

# Calculate the statistical properties of the time series
mean = data['Amount'].mean()
variance = data['Amount'].var()
autocorrelation = data['Amount'].autocorr()
st.write(f"Mean: {mean:.2f}")
st.write(f"Variance: {variance:.2f}")
st.write(f"Autocorrelation: {autocorrelation:.2f}")

st.write('Normal', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
st.write('Fraud', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

### Data Preprations
data["Time (second)"]=scale(data["Time (second)"])
data["Amount"]=scale(data["Amount"])
X_features=data.drop('Class',axis=1).columns
#'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15'], 


### n_cols is the number of features
n_cols=X_features.shape[0]

### X (independent and y targte varibale)
X=data[X_features].values
y=data["Class"].values

#SMOTE

#Split the dataset into train and tets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, stratify=y,test_size=.20,  train_size=0.8)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, train_size=0.8)
st.write("Shape of X_train Before SMOTE", (X_train.shape))

np.savetxt('Model_2_X_test.csv',X_test,delimiter =",")
np.savetxt('Model_2_Y_test.csv',y_test,delimiter =",")

#SMOTE the Train Portion
smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)
st.write("Shape of X_train After SMOTE", (X_train.shape))




X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val =  X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
numer_of_unique_values_of_target_varibale=len(np.unique(y_train))




### Initiate the model and form the model. It's convoluted network. 
### First layer is a convoluted layer with 40 output space and length of 10 in 1D convolution window
### then 10% of the info is forgotten and passed to the 2nd Layer with 35 nodes
### then 10% of the info is forgotten and passed to the 3rd Layer with 40 output space and length of 10 in 1D convolution window
### the 10% of the info is forgotten and passed to MaxPoolin to reduce dimentionality
### flatten will unstack all the tensor values into a 1-D tensor and teh passed 
### to a dense network having 2 outputs (Fruad or regular Transaction)


num_filters = 30
filter_size = 10
pool_size = 2

conv_1D_model = Sequential([
  Conv1D(num_filters, filter_size, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.1),
    Dense(30, activation='relu'),
    Conv1D(num_filters, filter_size, activation='relu'),
    Dropout(0.1),
  MaxPooling1D(pool_size=pool_size),
    Dropout(0.1),
  Flatten(),
    ### For the last layer the number of nodes has to be equal to the distinct number of target varibale  
  Dense(numer_of_unique_values_of_target_varibale, activation='sigmoid'),
])


conv_1D_model.compile(optimizer='adam',
  loss='binary_crossentropy', metrics=['accuracy'])

conv_1D_model_history=conv_1D_model.fit(
  X_train,
  to_categorical(y_train),
  epochs=1,#10
  validation_data=(X_test,
                   to_categorical(y_test)))

st.write('conv 1D model summary is: ')
st.write(conv_1D_model.summary())

# Visualizing loss and accuracy for Training set and validatio
training_loss = conv_1D_model_history.history['loss']
validation_loss = conv_1D_model_history.history['val_loss']
training_Acc = conv_1D_model_history.history['accuracy']
validation_Acc = conv_1D_model_history.history['val_accuracy']

# number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history for conv_1D_model
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss for 1D Convolutional', 'Validation Loss for 1D Convolutional'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
st.pyplot()


# Visualize accuracy history for conv_1D_model
plt.plot(epoch_count, training_Acc, 'r--')
plt.plot(epoch_count, validation_Acc, 'b-')
plt.legend(['Training Accuracy for 1D Convolutional', 'Validation Accuracy for 1D Convolutional'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
st.pyplot()


pickle.dump(conv_1D_model, open('Model_2.pkl', 'wb'))