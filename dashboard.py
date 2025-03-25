import streamlit as st
import pickle

# Load Model
model = pickle.load(open("lead_model.pkl", "rb"))

st.title("🚀 AI Lead Scoring Dashboard")

st.write("🎯 Predict & Track Lead Conversion Success!")

# Example Input
example_data = [[0.5, 0.2, -0.1, 1.3]]  # Replace with real data
prediction = model.predict(example_data)

st.write(f"🎯 Lead Conversion Probability: {prediction[0]}")

# Simulated Sales Leaderboard
sales_reps = {"Alice": 85, "Bob": 78, "Charlie": 92}
sorted_reps = sorted(sales_reps.items(), key=lambda x: x[1], reverse=True)

st.subheader("🏆 Sales Leaderboard")
for name, score in sorted_reps:
    st.write(f"{name}: {score} 🎉")
