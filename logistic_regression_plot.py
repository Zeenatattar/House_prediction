import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("data.csv")
data['expensive'] = (data['price'] > 500000).astype(int)

# Feature selection
X = data[['sqft_living', 'bathrooms']]
y = data['expensive']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Streamlit interface
st.title("🏠 House Price Classifier (Expensive vs Affordable)")
st.write("This model predicts if a house is expensive based on its square footage and number of bathrooms.")

# Input sliders
sqft = st.slider("Sqft Living", int(X['sqft_living'].min()), int(X['sqft_living'].max()), 2000)
baths = st.slider("Number of Bathrooms", float(X['bathrooms'].min()), float(X['bathrooms'].max()), 2.0, step=0.25)

# Predict
user_input = scaler.transform([[sqft, baths]])
prediction = model.predict(user_input)[0]
st.write(f"### 💡 Prediction: {'Expensive 🏡' if prediction else 'Affordable 🛏️'}")

# Plot decision boundary
st.subheader("Decision Boundary")

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolor='k')
ax.set_xlabel("Sqft Living (scaled)")
ax.set_ylabel("Bathrooms (scaled)")
st.pyplot(fig)
