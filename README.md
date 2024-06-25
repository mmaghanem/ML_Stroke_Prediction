# Stroke Prediction using Machine Learning and Deep Learning Techniques
This project aims to predict the likelihood of stroke based on health data using various machine learning and deep learning models. The project leverages Python, TensorFlow and other data science libraries to implement and compare different models to improve model accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling and Evaluation](#modeling-and-evaluation)
4. [Skills and Technologies Applied](#skills-and-technologies-applied)
5. [Benefits of the Project](#benefits-of-the-project)

## Introduction

Stroke is a leading cause of death and disability worldwide. This project aims to predict the likelihood of stroke using a dataset from Kaggle that contains various health-related attributes. We employ multiple machine learning and deep learning models, including Logistic Regression, Random Forest, and Keras Sequential models, to improve the prediction accuracy.

## Data Preprocessing

### Methods and Techniques

- **Data Cleaning**: Handle missing values and standardize data formats to ensure data quality.
- **Feature Engineering**: Convert categorical data to numerical using one-hot encoding and normalize numerical features.
- **Data Balancing**: Use techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('final_project_data.csv')

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']])

# Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])

# Combine all features
processed_data = pd.concat([pd.DataFrame(encoded_features), pd.DataFrame(scaled_features)], axis=1)

# Balance the dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(processed_data, data['stroke'])
```

## Modeling and Evaluation
### Logistic Regression
   - **Implementation:** Use sklearn.linear_model.LogisticRegression to build the model.
   - **Evaluation:** Assess the model using accuracy, precision, recall, and F1-score.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
```

### Random Forest
   - **Implementation:** Use sklearn.ensemble.RandomForestClassifier to build the model.
   - **Evaluation:** Assess the model using accuracy, precision, recall, and F1-score.
```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf = RandomForestClassifier(n_estimators=500, max_depth=8, bootstrap=False, max_features='auto')
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
```

### Deep Learning with Keras
   - **Implementation:** Use TensorFlow and Keras to build a sequential model.
   - **Evaluation:** Assess the model using accuracy, precision, recall, and F1-score.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Keras Sequential Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Predictions
y_pred_keras = (model.predict(X_test) > 0.5).astype("int32")
```

### Hyperparameter Tuning with Optuna
   - **Implementation:** Use Optuna to perform hyperparameter tuning for the Keras model.
   - **Evaluation:** Select the best model based on the optimization results.
```python
import optuna
from optuna.integration import TFKerasPruningCallback

def objective(trial):
    model = Sequential([
        Dense(trial.suggest_int('units1', 16, 128), activation=trial.suggest_categorical('activation1', ['relu', 'tanh']), input_shape=(X_train.shape[1],)),
        Dense(trial.suggest_int('units2', 16, 128), activation=trial.suggest_categorical('activation2', ['relu', 'tanh'])),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,
                        callbacks=[TFKerasPruningCallback(trial, 'val_accuracy')], verbose=0)
    
    return history.history['val_accuracy'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Skills and Technologies Applied
### Python Programming
   - **Data Analysis:** Used pandas for data manipulation and numpy for numerical computations.
   - **Machine Learning:** Implemented scikit-learn for building and evaluating machine learning models.
   - **Deep Learning:** Used TensorFlow and Keras for constructing and training deep learning models.
   - **Hyperparameter Tuning:** Applied Optuna for optimizing the hyperparameters of the deep learning models.

### Data Preprocessing
   - **Data Cleaning:** Addressed missing values and standardized data using KNNImputer, OneHotEncoder, and StandardScaler.
   - **Data Balancing:** Used SMOTE to handle class imbalance in the dataset.

### Model Evaluation
   - **Statistical Methods:** Evaluated models using accuracy, precision, recall, and F1-score.
   - **Visualization:** Employed matplotlib and seaborn to visualize data distributions and model performance.

## Benefits of the Project
1. **Enhanced Predictive Accuracy:**
   - Improved the accuracy of stroke prediction using advanced machine learning and deep learning techniques.
2. **Application of Advanced Python Skills:**
   - Demonstrated the practical application of Python in handling real-world data, performing statistical analysis, and building predictive models.
