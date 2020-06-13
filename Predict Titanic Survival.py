# Predict Titanic Survival
'''
Predict Titanic Survival
The RMS Titanic set sail on its maiden voyage in 1912, crossing the Atlantic from Southampton, England to New York City. The ship never completed the voyage, sinking to the bottom of the Atlantic Ocean after hitting an iceberg, bringing down 1,502 of 2,224 passengers onboard.

In this project we will create a Logistic Regression model that predicts which passengers survived the sinking of the Titanic, based on features like age and class.

The data we will be using for training our model is provided by Kaggle.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'male':0, 'female': 1})

# Fill the nan values in the age column
passengers['Age'].fillna(inplace=True, value = round(passengers['Age'].mean()))

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 1  else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 2  else 0)
print(passengers)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, test_features,train_labels, test_labels = train_test_split(features, survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Score the model on the train data
print(model.score(train_features, train_labels))

# Score the model on the test data
print(model.score(test_features, test_labels))
print(model.coef_)
# Analyze the coefficients


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,24.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
''' 
[chances of dying][chanced of surviving]
'''