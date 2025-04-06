import sys
print("Python Path:", sys.executable)
print("Python Version:", sys.version)

import requests

# Base URL for the API
BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:", response.json())
    assert response.status_code == 200

def test_predict():
    """Test the prediction endpoint."""
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Prediction Response:", response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_health()
    test_predict()
    print("All tests passed!") 