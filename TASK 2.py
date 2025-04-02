import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
np.random.seed(42)
num_samples = 1000

# Features: Age, Income, Previous Purchases, Website Visits
age = np.random.randint(18, 65, num_samples)
income = np.random.randint(20000, 100000, num_samples)
previous_purchases = np.random.randint(0, 20, num_samples)
website_visits = np.random.randint(1, 50, num_samples)

# Target Variable: Purchased (1 - Yes, 0 - No)
purchase_prob = 1 / (1 + np.exp(-(0.03 * age + 0.00002 * income + 0.1 * previous_purchases + 0.05 * website_visits - 5)))
purchased = np.random.binomial(1, purchase_prob, num_samples)

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Previous_Purchases": previous_purchases,
    "Website_Visits": website_visits,
    "Purchased": purchased
})

# Split data into train and test sets
X = df.drop(columns=["Purchased"])
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report
