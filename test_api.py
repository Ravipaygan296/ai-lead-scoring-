import requests

url = "http://127.0.0.1:8000/predict"
data = {"features": [0.5, 0.2, -0.1, 1.3]}  # Replace with actual features

response = requests.post(url, json=data)
print(response.json())
