#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:03:19 2024

@author: williamcheong
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from tensorflow.keras.layers import Dense, Dropout, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from collections import Counter
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import random as python_random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set seed for reproducibility
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# Load data
data = pd.read_excel('loan_data_ADA_assignment.xlsx')

# Columns to drop are id, date (it does not help us understand anything)
# Invoice column is being dropped too because it repeated. 
# Some of the columns like total_credit_rv has more than 10k NAs, which will cause loss of data if it is kept and NA is dropped later on.
columns_to_drop = [
    'id', 'member_id', 'last_pymnt_d', 'last_credit_pull_d',
    'mths_since_last_major_derog', 'mths_since_last_delinq', 'emp_title',
    'tot_coll_amt', 'tot_cur_bal', 'mths_since_last_record', 'funded_amnt_inv',
    'grade', 'pymnt_plan', 'desc', 'purpose', 'title', 'addr_state',
    'total_pymnt_inv', 'collection_recovery_fee', 'next_pymnt_d',
    'collections_12_mths_ex_med', 'policy_code', 'issue_d', 'earliest_cr_line',
    'emp_length', 'recoveries', 'total_rec_late_fee', 'total_credit_rv',
    'loan_status', 'verification_status'
]

# Drop the columns not useful for deep learning
data_cleaned = data.drop(columns=columns_to_drop)

# loan_is_bad, transform into numeric
data_cleaned['loan_is_bad'] = data_cleaned['loan_is_bad'].apply(lambda x: 0 if x == 0 else 1)

# Drop the xx behind the firdt three number so zip code can be use to train the model
data_cleaned['zip_code'] = data_cleaned['zip_code'].str.extract(r'(\d+)').astype(int)

# Transform different home_ownership into numeric using one_hot encoding, it is 0 if FALSE , 1 if TRUE
data_cleaned = pd.get_dummies(data_cleaned, columns=['home_ownership'], prefix='home')

# Nominal ordering for sub_grade. 
sub_grade_mapping = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5,
    "B1": 6, "B2": 7, "B3": 8, "B4": 9, "B5": 10,
    "C1": 11, "C2": 12, "C3": 13, "C4": 14, "C5": 15,
    "D1": 16, "D2": 17, "D3": 18, "D4": 19, "D5": 20,
    "E1": 21, "E2": 22, "E3": 23, "E4": 24, "E5": 25,
    "F1": 26, "F2": 27, "F3": 28, "F4": 29, "F5": 30,
    "G1": 31, "G2": 32, "G3": 33, "G4": 34, "G5": 35
}

# Mapped the sub_grade numeric values into the column sub_grade
data_cleaned['sub_grade'] = data_cleaned['sub_grade'].map(sub_grade_mapping)

# Remove NAs in revol_util
data_cleaned = data_cleaned.dropna(subset=['revol_util'])

# Dropped the unwanted extra home_columns
data_cleaned = data_cleaned.drop(['home_OTHER', 'home_NONE'], axis=1)

# Converting TRUE to 1 and FALSE to 0. 
data_cleaned['home_OWN'] = data_cleaned['home_OWN'].apply(lambda x: 0 if x == 0 else 1)

data_cleaned['home_RENT'] = data_cleaned['home_RENT'].apply(lambda x: 0 if x == 0 else 1)

data_cleaned['home_MORTGAGE'] = data_cleaned['home_MORTGAGE'].apply(lambda x: 0 if x == 0 else 1)

# Prepare features and labels, features are the variables to train the model, labels are the outcome of loan_is_bad
features = data_cleaned.drop(columns=['loan_is_bad']).values
labels = data_cleaned['loan_is_bad'].values
#print(features[0:20])
#print(labels[0:20])

# Since we need to use SMOTEENN, let's split using sklearn (could use TensorFlow but keeping as is for SMOTEENN compatibility)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features - TensorFlow's Normalization layer expects batches, so we normalize using sklearn here for simplicity
# Normalize all features in X_train for SMOTEENN application
normalizer = Normalization(axis=-1)  # Normalizes along the features axis
normalizer.adapt(X_train)  # Adapt to the training data

# Apply normalization to X_train and X_test, both training and testing variables (excluding outcome variable) needs to be normalised to reduce the range
X_train_normalized = normalizer(X_train)
X_test_normalized = normalizer(X_test)

# Apply SMOTEENN for both over-sampling and under-sampling on the scaled training data
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_normalized, y_train) #bothsampling on the training set for both the features and the label

# Convert the resampled data back into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled)).batch(64) 
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_normalized, y_test)).batch(64) # test set only normalised for the features and label remain unchanged

# Initial learning rate
initial_learning_rate = 0.001

# Define the learning rate schedule. Adjust the parameters as needed.
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

# Use the learning rate schedule in the optimizer.
optimizer = Adam(learning_rate=lr_schedule) 

# Model definition
model = Sequential([
    Dense(30, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_resampled.shape[1],)),
    Dropout(0.5),
    Dense(30, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(15, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(15, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(15, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives()])

# Train the model
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset) # the train_dataset contain x_resampled, and y_resampled namely the variables and outcome

# Model evaluation
evaluation = model.evaluate(test_dataset, verbose=2) 

# Extract and print metrics
accuracy, precision, recall, fn = evaluation[1], evaluation[2], evaluation[3], evaluation[4]
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, False Negatives: {fn}')

# Plotting the training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Lower epoch higher false negative cases. 

# Generate predictions for the confusion matrix and FNR calculation
predictions = model.predict(X_test_normalized) # This code fundamentally use the trained model to make predictions on the normalised test dataset)
predicted_classes = (predictions > 0.5).astype(int) # Converting the probabilities into class labels of 0 or 1. If probability > 0.5 then it is 1, 0 otherwise

test_features, test_labels = next(iter(test_dataset))

# Print out the predicted and the real values for comparison
for pred, real in zip(predicted_classes, test_labels):
    print(f"Predicted: {pred[0]};    Real: {real}")

# Generate confusion matrix
cm = confusion_matrix(y_test, predicted_classes) #Computes the confusion matrix using the actual result (y_test) and the predicted result from predicted_classes

# Calculate True Positives (TP) and False Negatives (FN)
TP = cm[1, 1] # true positive is at the 1st row and 1st column of cm 
FN = cm[1, 0]

# Calculate and print False Negative Rate (FNR)
FNR = FN / (FN + TP) 
print(f'False Negative Rate (Miss Rate): {FNR:.4f}')