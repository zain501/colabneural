#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:51:24 2024

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

# Load data
data = pd.read_excel('cleaned_data_23.xlsx')
data['zip_code'] = data['zip_code'].str.extract(r'(\d+)').astype(int)

# Prepare features and labels
features = data.drop(columns=['loan_is_bad']).values
labels = data['loan_is_bad'].values

# Split the data into training and testing before resampling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize all features in X_train for SMOTEENN application
normalizer = Normalization(axis=-1)  # Normalizes along the features axis
normalizer.adapt(X_train)  # Adapt to the training data

# Apply normalization to X_train and X_test
X_train_normalized = normalizer(X_train)
X_test_normalized = normalizer(X_test)

# Apply SMOTEENN for both over-sampling and under-sampling on the normalized training data
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_normalized, y_train)

class_distribution = Counter(y_resampled)
print(class_distribution)

# Proceed with model definition, compilation, and training as before, using X_resampled and y_resampled
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_resampled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives()])

# Since X_resampled is already a NumPy array from the SMOTEENN process, we can directly use it in model.fit
model.fit(X_resampled, y_resampled, epochs=100, batch_size=128)

# Evaluate the model on the normalized test set
evaluation = model.evaluate(X_test_normalized, y_test)

# Extract and print metrics
accuracy, precision, recall, fn = evaluation[1], evaluation[2], evaluation[3], evaluation[4]
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, False Negatives: {fn}')
