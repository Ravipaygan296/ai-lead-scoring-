import shap
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
X_train, X_test, y_train, y_test = pickle.load(open("data.pkl", "rb"))

# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Explainability: Use SHAP to show feature importance
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Save model & explanations
pickle.dump(model, open("lead_model.pkl", "wb"))
pickle.dump(shap_values, open("shap_values.pkl", "wb"))
