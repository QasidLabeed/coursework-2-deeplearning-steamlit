import pandas as pd
import sklearn.metrics as metrique
from pandas import Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import LSTM, Dense, Embedding, Dropout,Input, Attention, Layer, Concatenate, Permute, Dot, Multiply, Flatten
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import Sequential
from keras import backend as K, regularizers, Model, metrics
from keras.backend import cast
import streamlit as st 



st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv('creditcard.csv', na_filter=True)
col_del = ['Time (second)' ,'V5', 'V6', 'V7', 'V8', 'V9','V13','V15', 'V16',  'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
tr_data = data.drop(col_del,axis =1)
tr_data.shape

X = tr_data.drop(['Class'], axis = 'columns')
Label_Data = tr_data['Class']

# Generate and plot imbalanced classification dataset
from collections import Counter
from matplotlib import pyplot
from numpy import where
# summarize class distribution
counter = Counter(tr_data['Class'])
st.write(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(tr_data['Class'] == label)[0]
	
 # transform the dataset
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_r, y = oversample.fit_resample(X, tr_data['Class'])
# summarize the new class distribution
counter = Counter(y)
st.write(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	
from sklearn.preprocessing import StandardScaler
## Standardizing the data
X_r2 = StandardScaler().fit_transform(X_r)


# Splitting Data

from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(X_r2,y,test_size=0.2,train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)



x_train.shape



x_test.shape


x_val.shape



# design network
np.random.seed(7)


train_LSTM_X=x_train
val_LSTM_X=x_test

train_LSTM_X = train_LSTM_X.reshape((train_LSTM_X.shape[0], 1, train_LSTM_X.shape[1]))
val_LSTM_X = val_LSTM_X.reshape((val_LSTM_X.shape[0], 1, val_LSTM_X.shape[1]))


train_LSTM_y=y_train
val_LSTM_y=y_test

# add a Reshape layer after the first LSTM layer to convert the output to a 3D tensor
# before feeding it to the second LSTM layer

from keras.layers import Input, LSTM, Dense, Reshape
from keras.models import Model

inputs=Input((1,8))
x1 = LSTM(50,dropout=0.2,recurrent_dropout=0.2)(inputs)
x2 = Reshape((1,50))(x1)
x3 = LSTM(50,dropout=0.2,recurrent_dropout=0.2)(x2)
outputs = Dense(1,activation='sigmoid')(x3)
model = Model(inputs,outputs)




# wrapping the output of the first LSTM layer in a Reshape

from keras.layers import Input, LSTM, Dense, Reshape
from keras.models import Model

inputs=Input((1,9))
x1 = LSTM(50,dropout=0.2,recurrent_dropout=0.2)(inputs)
x2 = Reshape((1,50))(x1)  # add Reshape layer here
x3 = LSTM(50,dropout=0.2,recurrent_dropout=0.2)(x2)
outputs = Dense(1,activation='sigmoid')(x3)
model = Model(inputs,outputs)



train_LSTM_y = y_train
val_LSTM_y = y_test

## create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, input_shape=(train_LSTM_X.shape[1], train_LSTM_X.shape[2])))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_LSTM_X, train_LSTM_y, epochs=1, batch_size=72, validation_data=(val_LSTM_X, val_LSTM_y), verbose=2, shuffle=False)

# save model and architecture to single file
model.save('Save_Model.h5')
st.write("Saved model to disk")




# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
 
# load model
model = load_model('Save_Model.h5')
# summarize model.
model.summary()



import matplotlib.pyplot as pyplot

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

st.pyplot()

st.write()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()

st.pyplot()




loss = model.evaluate(train_LSTM_X, train_LSTM_y, verbose=0)
st.write('Train Loss:', loss)




# predict probabilities for test set
yhat_probs = model.predict(val_LSTM_X, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]


# output of the LSTM model is a continuous value, regression metrics is used for evaluating performance of the model

from sklearn.metrics import mean_absolute_error, mean_squared_error

# calculate the mean absolute error (MAE)
mae = mean_absolute_error(val_LSTM_y, yhat_probs)
st.write('MAE: %f' % mae)

# calculate the mean squared error (MSE)
mse = mean_squared_error(val_LSTM_y, yhat_probs)
st.write('MSE: %f' % mse)



# accuracy, precision, recall, and F1-score, are designed to work with binary or categorical targets but LSTM has continuous output.
#we have mix of binary and continuous targets, so converting continuous target values to binary labels by setting a threshold value

threshold = 0.5
yhat_classes = np.where(yhat_probs > threshold, 1, 0)


# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(val_LSTM_y, yhat_classes)
st.write('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(val_LSTM_y, yhat_classes)
st.write('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(val_LSTM_y, yhat_classes)
st.write('Recall: %f' % recall)

 
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=val_LSTM_y, y_pred=yhat_classes)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function st.writes and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    st.write(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
labels = ['Normal','Fraud']


plot_confusion_matrix(cm=cm, classes=labels, title='LSTM')    


#attention mechanism used to weigh input features based on their importance.

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

      
#two LSTM layers with dropout, followed by a custom attention layer, and a fully connected layer with a sigmoid activation function for binary classification

inputs1=Input((1,9))
att_in=LSTM(50,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(inputs1)
att_in_1=LSTM(50,return_sequences=True,dropout=0.3,recurrent_dropout=0.2)(att_in)
att_out=attention()(att_in_1)
outputs1=Dense(1,activation='sigmoid',trainable=True)(att_out)
model1=Model(inputs1,outputs1)

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history1 = model1.fit(train_LSTM_X, train_LSTM_y, epochs=1, batch_size=30000, validation_data=(val_LSTM_X, val_LSTM_y))
      
# save Attention model and architecture to single file
model1.save('Save_Model_Attention.h5')
st.write("Saved model to disk")        
  
        
from keras.models import load_model
import keras


# st.write a summary of the loaded model
model1.summary()


# evaluate the model
_, train_acc = model1.evaluate(train_LSTM_X, train_LSTM_y, verbose=0)
_, test_acc = model1.evaluate(val_LSTM_X, val_LSTM_y, verbose=0)
st.write('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# predict probabilities for test set
yhat_probs1 = model1.predict(val_LSTM_X, verbose=0)
# reduce to 1d array
yhat_probs1 = yhat_probs1[:, 0]


import numpy as np

# apply threshold of 0.5 to convert probabilities to binary predictions
yhat1 = np.where(yhat_probs1 > 0.5, 1, 0)

# calculate classification metrics
accuracy = accuracy_score(val_LSTM_y, yhat1)
precision = precision_score(val_LSTM_y, yhat1)
recall = recall_score(val_LSTM_y, yhat1)
confusion_mat = confusion_matrix(val_LSTM_y, yhat1)


# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(val_LSTM_y, yhat1)
st.write('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(val_LSTM_y, yhat1)
st.write('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(val_LSTM_y, yhat1)
st.write('Recall: %f' % recall)


cm1 = confusion_matrix(y_true=val_LSTM_y, y_pred=yhat1)

      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 
        

































