import requests

response = requests.get("http://localhost:5000/health")
print(response.json())
