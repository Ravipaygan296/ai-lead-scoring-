import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset (Replace with actual dataset)
df = pd.read_csv("leads_data.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Feature Engineering: Create useful insights
df["interaction_score"] = df["page_views"] * df["time_spent"]  
df["engagement_rate"] = df["clicks"] / (df["emails_sent"] + 1)

# Balance dataset (Handling Bias)
smote = SMOTE()
X, y = df.drop(columns=["converted"]), df["converted"]
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale Data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Save for later use
pickle.dump((X_train, X_test, y_train, y_test), open("data.pkl", "wb"))
