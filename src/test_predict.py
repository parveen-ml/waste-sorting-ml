import requests

url = "http://127.0.0.1:5000/predict"
file_path = "D:/Projects_Repo/waste-sorting-ml/data/test/glass/glass_454.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Prediction:", response.text)
