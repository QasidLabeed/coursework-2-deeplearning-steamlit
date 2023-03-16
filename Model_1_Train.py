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


st.set_option('deprecation.showPyplotGlobalUse', False)

color_palette = sns.color_palette()
plt.style.use('seaborn-pastel')

# Load the Credit Card Fraud detection dataset
df = pd.read_csv(r'creditcard.csv')


########################### Data Preparation ###############################

## Convert Time in seconds to datetime (This will start the time from 1970 since the timestamp starts from zero)
## But we can ignore it since we will only consider 2 days relatively to the start (zero timestamp)
df['Time (second)'] = pd.to_datetime(df['Time (second)'], unit='s')

## Set the Time as index
df = df.set_index('Time (second)')

## Add hour column in dataframe for plotting
df['hour'] = df.index.hour

## As most of our data is already scaled and time will be dropped so only scalling Amount
rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))


## Data Analysis
df.describe()

## Plot all the transactions by Index
st.write("## Credit Card Fraud Transactions")
st.line_chart(df)

## Plot Transaction Amounts each Hour
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y="Amount")
ax.set_title("Transaction Amounts each hour ")
st.pyplot(fig)

## Drop Amount since now scaled_amount will be used
df = df.drop('Amount',axis=1)

## Drop the hour column since it only used for plotting
df = df.drop('hour',axis=1)

## Correlation between features
f, ax1 = plt.subplots(figsize=(24,20))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
st.pyplot(f)

# Split the dataset into training, validation, and test sets
train = df.loc[df.index < '01-02-1970 12:00:00']

test = df.loc[(df.index >= '01-02-1970 12:00:00') & (df.index < '01-02-1970 18:00:00')]

validate = df.loc[df.index >= '01-02-1970 18:00:00']

## Training Data
X_train = train.drop('Class',axis=1)
y_train = train['Class']

## Validation Data
X_val = validate.drop('Class',axis=1)
y_val = validate['Class']

## Testing Data
X_test = test.drop('Class',axis=1)
X_test = X_test.values

y_test = test['Class']
y_test = y_test.values

np.savetxt('Model_1_X_test.csv',X_test,delimiter =",")
np.savetxt('Model_1_Y_test.csv',y_test,delimiter =",")




## Plot Training, Test and Validation Data
fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax, label='Training Set', title="Data Train/Test Split")
ax.axvline('01-02-1970 12:00:00', color='black', ls='--')
test.plot(ax=ax, label='Test set')
ax.axvline('01-02-1970 18:00:00', color='green', ls='--')
validate.plot(ax=ax, label="Validation set")
ax.legend(['Training Set', 'Test Set', 'Validation Set'])
plt.show()


# Define the input shape of the model based on the number of features in the dataset
input_shape = (X_train.shape[1],)


# Create a sequential model in Keras with dense layers that process the input features
sequentialModel = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
# Compile the model with binary cross-entropy loss and Adam optimizer
sequentialModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data and evaluate its performance on the validation data
sequentialModel.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10, batch_size=128
)

pickle.dump(sequentialModel, open('Model_1.pkl', 'wb'))