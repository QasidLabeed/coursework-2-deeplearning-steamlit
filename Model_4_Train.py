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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pickle

 

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

st.set_option('deprecation.showPyplotGlobalUse', False)
#Loading the data

df = pd.read_csv('creditcard.csv')
st.write('df.head()')
st.write(df.head())

# plotting the data

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
st.pyplot()

#Checking the two types of transaction in the dataset

frauds = df[df.Class == 1]
normal = df[df.Class == 0]

st.write(frauds.shape)
st.write(normal.shape)

# checking different  amount of money used in different transaction classes

st.write('Fraud Amount description')
st.write(frauds.Amount.describe())

st.write('Normal Amount description')
st.write(normal.Amount.describe())

# visualization of both type of transactions:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 30

ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
st.pyplot()

# Check if fraud transaction occurs at certain time:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds['Time (second)'], frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal['Time (second)'], normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
st.pyplot()

# Check if fraud transaction occurs at certain time:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds['Time (second)'], frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal['Time (second)'], normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
st.pyplot()

# The classes are heavily skewed we need to solve this issue later.
st.write('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
st.write('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=10)

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time (second)'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=10)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=10)
ax[1].set_xlim([min(time_val), max(time_val)])


st.pyplot()

#scale the columns comprise of Time and Amount . Time and amount should be scaled as the other columns.
#we need to also create a sub sample of the dataframe in order to have an equal amount of Fraud and Non-Fraud cases, 

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time (second)'].values.reshape(-1,1))

df.drop(['Time (second)','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

st.write('Amount and Time are Scaled!')

df.head()

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

st.write('Distribution of the Classes in the subsample dataset')
st.write(new_df['Class'].value_counts()/len(new_df))



sns.countplot('Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
st.pyplot()

# Make sure we use the subsample in our correlation

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
st.pyplot()

# Autoencoder Neural Network (in Keras) 
# Splitting the data into 3 parts for training,testing and validation in the ratio 60:20:20 :

from sklearn.model_selection import train_test_split

x, y = new_df.iloc[:, :-1], new_df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, train_size=0.75)

np.savetxt('Model_4_X_test.csv',x_test,delimiter =",")
np.savetxt('Model_4_Y_test.csv',y_test,delimiter =",")

# Building model: auto encoder contains 4 layes: 2 for encoding and 2 for decoding:

input_dim = x_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# training model for 100 epochs with a batch size of 20 samples to have best performing model:

nb_epoch = 100
batch_size = 20

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(x_train, x_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


pickle.dump(autoencoder, open('Model_4.pkl', 'wb'))