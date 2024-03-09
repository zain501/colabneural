#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:01:32 2024

@author: williamcheong
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler
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
data = pd.read_excel('cleaned_data_23.xlsx')
data['zip_code'] = data['zip_code'].str.extract(r'(\d+)').astype(int)

# Prepare features and labels
features = data.drop(columns=['loan_is_bad']).values
labels = data['loan_is_bad'].values

# Since we need to use SMOTEENN, let's split using sklearn (could use TensorFlow but keeping as is for SMOTEENN compatibility)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features - TensorFlow's Normalization layer expects batches, so we normalize using sklearn here for simplicity
# Normalize all features in X_train for SMOTEENN application
normalizer = Normalization(axis=-1)  # Normalizes along the features axis
normalizer.adapt(X_train)  # Adapt to the training data

# Apply normalization to X_train and X_test
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
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_resampled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives()])

# Train the model
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

# Model evaluation
evaluation = model.evaluate(test_dataset)

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
predictions = model.predict(X_test_normalized)
predicted_classes = (predictions > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_test, predicted_classes)

# Calculate True Positives (TP) and False Negatives (FN)
TP = cm[1, 1]
FN = cm[1, 0]

# Calculate and print False Negative Rate (FNR)
FNR = FN / (FN + TP)
print(f'False Negative Rate (Miss Rate): {FNR:.4f}')
