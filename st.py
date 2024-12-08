import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

# Load and preprocess the data
data = pd.read_csv('AI/final/breast-cancer.csv')
encoder = LabelEncoder()
data['diagnosis'] = encoder.fit_transform(data['diagnosis'])
data = data.drop(columns=['id'])
X = data.drop(columns='diagnosis')
Y = data['diagnosis']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Build the model
tf.random.set_seed(3)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check if model training history is None
history = None

try:
    history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
except Exception as e:
    st.error(f"Error during model training: {str(e)}")
    st.stop()

# Prediction logic
def main():
    st.title("Breast Cancer Prediction Model")

    # Input fields for features
    feature_names = list(X.columns)
    inputs = {}

    for feature in feature_names:
        inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Prediction button
    if st.button("Predict"):
        # Convert input data to numpy array and reshape
        input_data = np.array([inputs[feature] for feature in feature_names])
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize input data
        input_data_std = scaler.transform(input_data_reshaped)

        # Predict
        prediction = model.predict(input_data_std)
        prediction_label = np.argmax(prediction)

        if prediction_label == 0:
            st.write('The tumor is Malignant')
        else:
            st.write('The tumor is Benign')

if __name__ == '__main__':
    main()
