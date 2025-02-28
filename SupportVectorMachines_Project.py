import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Load Dataset
iris = sns.load_dataset('iris')
print(iris.head())    # Display first few rows


# Exploratory Data Analysis (EDA)
# Pairplot of the dataset
sns.pairplot(iris, hue='species', palette='Dark2')
plt.show()

# KDE plot: Sepal Width vs. Sepal Length for Setosa species
setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(x=setosa['sepal_width'], y=setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)
plt.title('KDE Plot of Setosa Sepal Width vs. Sepal Length')
plt.show()


# Train-Test Split
# Split data into a training set and testing set
X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Training a Support Vector Classifier
# Call the model and fit the model to the training data
svc_model = SVC()
svc_model.fit(X_train, y_train)


# Model Evaluation
# Get predictions
predictions = svc_model.predict(X_test)

# Classification Report and Confusion Matrix
print("\nSVC Model Performance:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# Hyperparameter Tuning with GridSearchCV
# Define hyperparameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

# Perform GridSearch
grid = GridSearchCV(SVC(), param_grid, verbose=1)
grid.fit(X_train, y_train)

# Get best parameters
print("\nBest Parameters from GridSearch:", grid.best_params_)

# Make predictions using the best model
grid_predictions = grid.predict(X_test)

# Print classification report and confusion matrix for GridSearch model
print("\nOptimized SVC Model Performance:")
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
